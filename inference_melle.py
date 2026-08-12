"""Autoregressive MELLE continuation inference with a 24 kHz Vocos decoder."""

from __future__ import annotations

import argparse
import os

import torch
import torchaudio

from melle_dataset import MelConfig
from melle_model import MelleModel, MelleModelArgs
from melle_tokenizer import MelleCharacterTokenizer
from vocos.pretrained import Vocos


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="melle_character_tokenizer.model")
    parser.add_argument(
        "--prompt-audio",
        default="",
        help="Optional reference audio; omit it to synthesize directly from text",
    )
    parser.add_argument(
        "--prompt-copy-output",
        default="",
        help="Optional Vocos copy-synthesis WAV used to verify the prompt feature pipeline",
    )
    parser.add_argument(
        "--text",
        required=True,
        help="Target text, or prompt transcript plus target text when prompt audio is supplied",
    )
    parser.add_argument("--output", default="melle_output.wav")
    parser.add_argument("--vocos-repo", default="charactr/vocos-mel-24khz")
    parser.add_argument("--max-new-seconds", type=float, default=10.0)
    parser.add_argument("--min-new-seconds", type=float, default=0.25)
    parser.add_argument("--stop-threshold", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def load_prompt(path, config, feature_extractor, device):
    waveform, source_rate = torchaudio.load(path)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if source_rate != config.sample_rate:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=source_rate, new_freq=config.sample_rate
        )
    if waveform.size(1) == 0:
        raise ValueError(f"prompt audio is empty: {path}")
    waveform = waveform.to(device)
    features = feature_extractor(waveform).squeeze(0).transpose(0, 1)
    return features.contiguous()


@torch.inference_mode()
def generate(model, text_ids, prompt, config, args):
    device = prompt.device
    text_ids = text_ids.unsqueeze(0).to(device)
    text_mask = torch.ones_like(text_ids, dtype=torch.bool)
    known_mel = prompt
    prompt_steps = prompt.size(0)
    min_steps = max(1, round(args.min_new_seconds * config.sample_rate / config.hop_length))
    max_steps = max(min_steps, round(args.max_new_seconds * config.sample_rate / config.hop_length))
    # A full-length final context can still predict one additional frame.
    available_steps = model.args.max_seq_len - text_ids.size(1) - prompt_steps + 1
    if available_steps < min_steps:
        raise ValueError(
            "text and prompt leave insufficient model context for the requested "
            f"minimum generation: available={available_steps}, required={min_steps} frames"
        )
    max_steps = min(max_steps, available_steps)

    for step in range(max_steps):
        # Training predicts y[t] from y[t-1]. The supplied acoustic prompt
        # provides the initial context, so inference needs no synthetic GO.
        mel_inputs = known_mel.unsqueeze(0)
        mel_mask = torch.ones(
            1, mel_inputs.size(1), dtype=torch.bool, device=device
        )
        outputs = model(
            text_ids,
            text_mask,
            mel_inputs,
            mel_mask,
            sample_latent=True,
            apply_postnet=False,
        )
        next_frame = outputs["coarse_mel"][0, -1].float()
        stop_probability = outputs["stop_logits"][0, -1].float().sigmoid().item()
        known_mel = torch.cat([known_mel, next_frame.unsqueeze(0)], dim=0)
        if step + 1 >= min_steps and stop_probability >= args.stop_threshold:
            break

    # Apply the causal post-net once after coarse AR generation concludes.
    coarse = known_mel.unsqueeze(0)
    model_dtype = next(model.parameters()).dtype
    refined = model.refine_mel(coarse.to(model_dtype)).float()
    # The supplied prompt is already ground-truth Vocos feature context.
    # Preserve it exactly and apply post-net refinement only to generated
    # frames in the returned prompt-plus-continuation sequence.
    refined[:, :prompt_steps] = coarse[:, :prompt_steps]
    return refined[0], prompt_steps


def main():
    args = parse_args()
    if not os.path.exists(args.tokenizer):
        raise FileNotFoundError(f"tokenizer not found: {args.tokenizer}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed(args.seed)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_args = MelleModelArgs(**checkpoint["model_args"])
    config = MelConfig(**checkpoint["mel_config"])
    model = MelleModel(model_args)
    model.load_state_dict(checkpoint["model"])
    model.to(device=device, dtype=torch.bfloat16 if device == "cuda" else torch.float32).eval()

    tokenizer = MelleCharacterTokenizer(args.tokenizer)
    if not tokenizer.processor.load(args.tokenizer):
        raise ValueError(f"failed to load tokenizer: {args.tokenizer}")
    token_values = [tokenizer.bos_token_id]
    token_values.extend(tokenizer.encode(args.text))
    token_values.append(tokenizer.eos_token_id)
    text_ids = torch.tensor(token_values, dtype=torch.long)

    vocos = Vocos.from_pretrained(args.vocos_repo).to(device).eval()
    if args.prompt_audio:
        prompt = load_prompt(args.prompt_audio, config, vocos.feature_extractor, device)
    else:
        prompt = torch.empty(
            0,
            model_args.mel_dim,
            device=device,
            dtype=torch.float32,
        )
    if args.prompt_copy_output and not args.prompt_audio:
        raise ValueError("--prompt-copy-output requires --prompt-audio")
    if args.prompt_copy_output:
        prompt_audio = vocos.decode(prompt.transpose(0, 1).unsqueeze(0)).cpu()
        torchaudio.save(args.prompt_copy_output, prompt_audio, config.sample_rate)
        print(f"Saved Vocos prompt copy: {args.prompt_copy_output}")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        features, prompt_steps = generate(model, text_ids, prompt, config, args)
    generated_features = features[prompt_steps:]
    prompt_stats = (
        f"prompt mean={prompt.mean().item():.3f}, std={prompt.std().item():.3f}; "
        if prompt_steps
        else "prompt=none; "
    )
    print(
        "Feature statistics: "
        f"{prompt_stats}"
        f"generated mean={generated_features.mean().item():.3f}, "
        f"std={generated_features.std().item():.3f}, "
        f"min={generated_features.min().item():.3f}, "
        f"max={generated_features.max().item():.3f}"
    )
    audio = vocos.decode(features.transpose(0, 1).unsqueeze(0).to(device)).cpu()
    torchaudio.save(args.output, audio, config.sample_rate)
    print(
        f"Saved {args.output}: prompt={prompt_steps} frames, "
        f"generated={features.size(0) - prompt_steps} frames, "
        f"sample_rate={config.sample_rate}"
    )


if __name__ == "__main__":
    main()
