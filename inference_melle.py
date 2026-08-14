"""Autoregressive MELLE continuation inference with a 24 kHz Vocos decoder."""

from __future__ import annotations

import argparse
import os
import re
import unicodedata

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
    parser.add_argument(
        "--seed-search",
        type=int,
        default=1,
        help="Number of consecutive seeds to evaluate; requires --asr-model",
    )
    parser.add_argument(
        "--asr-model",
        default="",
        help=(
            "Optional ASR model; with faster-whisper, use a size such as "
            "tiny.en; enabling this prints sCER/WER for the generated "
            "continuation"
        ),
    )
    parser.add_argument(
        "--asr-backend",
        choices=("faster-whisper", "transformers"),
        default="faster-whisper",
        help="ASR backend used for sCER/WER scoring (default: faster-whisper)",
    )
    parser.add_argument(
        "--asr-reference",
        default="",
        help="Reference text for ASR scoring; defaults to --text",
    )
    parser.add_argument(
        "--asr-language",
        default="en",
        help="Whisper language code passed to ASR scoring (default: en)",
    )
    return parser.parse_args()


def _edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for row, reference_item in enumerate(reference, start=1):
        current = [row]
        for column, hypothesis_item in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (reference_item != hypothesis_item),
                )
            )
        previous = current
    return previous[-1]


def _normalize_transcript(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(
        character
        for character in text
        if not unicodedata.category(character).startswith("P")
        and not character.isspace()
    )


def _normalize_words(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    return re.findall(r"[\w]+", text, flags=re.UNICODE)


def load_asr_model(model_name: str, device: str, backend: str):
    if backend == "faster-whisper":
        try:
            from faster_whisper import WhisperModel
        except ImportError as error:
            raise RuntimeError(
                "--asr-backend faster-whisper requires the 'faster-whisper' "
                "package; install it first"
            ) from error

        asr_device = "cpu"
        compute_type = "int8"
        if device == "cuda":
            try:
                import ctranslate2

                cuda_types = ctranslate2.get_supported_compute_types("cuda")
            except (ImportError, RuntimeError, ValueError):
                cuda_types = set()
            if cuda_types:
                asr_device = "cuda"
                compute_type = next(
                    compute
                    for compute in ("float16", "int8_float16", "int8_float32", "int8")
                    if compute in cuda_types
                )
            else:
                print(
                    "faster-whisper: CTranslate2 CUDA support is unavailable; "
                    "using CPU int8 for ASR scoring"
                )
        return WhisperModel(
            model_name,
            device=asr_device,
            compute_type=compute_type,
        )

    try:
        from transformers import pipeline
    except ImportError as error:
        raise RuntimeError(
            "--asr-backend transformers requires the 'transformers' package; "
            "install requirements.txt first"
        ) from error

    asr_device = 0 if device == "cuda" else -1
    return pipeline(
        "automatic-speech-recognition",
        model=model_name,
        device=asr_device,
    )


def score_generated_audio(
    audio: torch.Tensor,
    reference_text: str,
    sample_rate: int,
    model_name: str,
    language: str,
    device: str,
    backend: str = "faster-whisper",
) -> dict:
    """Transcribe generated audio and return space-insensitive CER/WER."""
    if not model_name:
        raise ValueError("model_name is required for ASR scoring")
    recognizer = load_asr_model(model_name, device, backend)
    return transcribe_and_score(
        recognizer, audio, reference_text, sample_rate, language, backend
    )


def transcribe_and_score(
    recognizer,
    audio: torch.Tensor,
    reference_text: str,
    sample_rate: int,
    language: str,
    backend: str = "faster-whisper",
) -> dict:
    waveform = audio.squeeze(0).numpy()
    if backend == "faster-whisper":
        segments, _ = recognizer.transcribe(
            waveform,
            language=language or None,
            task="transcribe",
            beam_size=1,
            vad_filter=False,
        )
        hypothesis = "".join(segment.text for segment in segments).strip()
    else:
        recognizer_input = {"raw": waveform, "sampling_rate": sample_rate}
        generate_kwargs = {"task": "transcribe"}
        if language:
            generate_kwargs["language"] = language
        result = recognizer(recognizer_input, generate_kwargs=generate_kwargs)
        hypothesis = result["text"].strip()

    reference_chars = _normalize_transcript(reference_text)
    hypothesis_chars = _normalize_transcript(hypothesis)
    reference_words = _normalize_words(reference_text)
    hypothesis_words = _normalize_words(hypothesis)
    scer = _edit_distance(reference_chars, hypothesis_chars) / max(1, len(reference_chars))
    wer = _edit_distance(reference_words, hypothesis_words) / max(1, len(reference_words))
    return {
        "reference": reference_text,
        "hypothesis": hypothesis,
        "scer": scer,
        "wer": wer,
    }


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


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

    # The paper applies the non-causal post-net only after coarse AR generation
    # concludes, so refined frames are never fed back into autoregressive input.
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
    if args.seed_search < 1:
        raise ValueError("--seed-search must be at least 1")
    if args.seed_search > 1 and not args.asr_model:
        raise ValueError("--seed-search requires --asr-model for best-seed selection")
    if args.asr_model and args.prompt_audio and not args.asr_reference:
        raise ValueError(
            "--asr-reference is required with --prompt-audio so the prompt "
            "transcript is not scored as generated speech"
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
    candidates = []
    for offset in range(args.seed_search):
        candidate_seed = args.seed + offset
        set_seed(candidate_seed)
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"
        ):
            features, prompt_steps = generate(model, text_ids, prompt, config, args)
        features = features.detach()
        generated_features = features[prompt_steps:]
        audio = vocos.decode(features.transpose(0, 1).unsqueeze(0).to(device)).cpu()
        generated_audio = (
            vocos.decode(
                generated_features.transpose(0, 1).unsqueeze(0).to(device)
            ).cpu()
            if prompt_steps
            else audio
        )
        candidates.append(
            {
                "seed": candidate_seed,
                "features": features.cpu(),
                "prompt_steps": prompt_steps,
                "audio": audio,
                "generated_audio": generated_audio,
            }
        )

    best = candidates[0]
    if args.asr_model:
        # Release the large MELLE/Vocos modules before loading Whisper when
        # scoring on GPU; candidate audio/features are already on CPU.
        del model, vocos
        if device == "cuda":
            torch.cuda.empty_cache()
        recognizer = load_asr_model(args.asr_model, device, args.asr_backend)
        reference = args.asr_reference or args.text
        for candidate in candidates:
            result = transcribe_and_score(
                recognizer,
                candidate["generated_audio"],
                reference,
                config.sample_rate,
                args.asr_language,
                args.asr_backend,
            )
            candidate["asr"] = result
            print(
                f"seed={candidate['seed']} sCER={result['scer']:.4f} "
                f"WER={result['wer']:.4f}: {result['hypothesis']}"
            )
        best = min(candidates, key=lambda candidate: candidate["asr"]["scer"])
        print(
            f"Selected seed={best['seed']} by minimum sCER="
            f"{best['asr']['scer']:.4f}"
        )

    features = best["features"]
    prompt_steps = best["prompt_steps"]
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
    torchaudio.save(args.output, best["audio"], config.sample_rate)
    print(
        f"Saved {args.output}: seed={best['seed']}, prompt={prompt_steps} frames, "
        f"generated={features.size(0) - prompt_steps} frames, "
        f"sample_rate={config.sample_rate}"
    )
    if args.asr_model:
        result = best["asr"]
        print(f"ASR reference: {result['reference']}")
        print(f"ASR hypothesis: {result['hypothesis']}")
        print(f"ASR sCER: {result['scer']:.4f} ({result['scer'] * 100:.2f}%)")
        print(f"ASR WER: {result['wer']:.4f} ({result['wer'] * 100:.2f}%)")


if __name__ == "__main__":
    main()
