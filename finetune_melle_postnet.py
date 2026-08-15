"""Fine-tune only MELLE's post-net on detached autoregressive rollouts."""

from __future__ import annotations

import argparse
import os
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from melle_dataset import MelConfig, MelleDataset, melle_collate_fn
from melle_model import MelleModel, MelleModelArgs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune MELLE's post-net on inference-like AR coarse mels"
    )
    parser.add_argument("--checkpoint", required=True, help="Stage 1 checkpoint")
    parser.add_argument("--manifest", default="manifest.json")
    parser.add_argument("--tokenizer", default="melle_character_tokenizer.model")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--max-iters", type=int, default=50_000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument(
        "--sample-latent",
        action="store_true",
        help="Sample latent frames during rollout instead of using their means",
    )
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def cycle(loader):
    while True:
        yield from loader


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.no_grad()
def autoregressive_rollout(model, batch, rollout_steps, sample_latent):
    """Generate coarse continuation frames without retaining the LM graph."""
    prompt_steps = batch["prompt_steps"]
    target_lengths = batch["mel_target_mask"].sum(dim=1)
    steps = min(
        rollout_steps,
        max(0, int((target_lengths - prompt_steps).min().item())),
    )
    if steps < 1:
        return None, None

    known_mels = [
        batch["mel_targets"][row, : int(prompt_steps[row].item())]
        for row in range(batch["mel_targets"].size(0))
    ]
    for _ in range(steps):
        max_len = max(mel.size(0) for mel in known_mels)
        mel_inputs = batch["mel_targets"].new_zeros(
            len(known_mels), max_len, batch["mel_targets"].size(-1)
        )
        mel_mask = torch.zeros(
            len(known_mels), max_len, dtype=torch.bool, device=mel_inputs.device
        )
        for row, mel in enumerate(known_mels):
            mel_inputs[row, : mel.size(0)] = mel
            mel_mask[row, : mel.size(0)] = True

        outputs = model(
            batch["text_ids"],
            batch["text_mask"],
            mel_inputs,
            mel_mask,
            prompt_lengths=prompt_steps,
            sample_latent=sample_latent,
            apply_postnet=False,
        )
        indices = mel_mask.sum(dim=1)
        next_frames = outputs["coarse_mel"][
            torch.arange(len(known_mels), device=mel_inputs.device), indices
        ].float()
        known_mels = [
            torch.cat([mel, next_frames[row : row + 1].to(mel.dtype)], dim=0)
            for row, mel in enumerate(known_mels)
        ]

    max_len = max(mel.size(0) for mel in known_mels)
    rollout = batch["mel_targets"].new_zeros(
        len(known_mels), max_len, batch["mel_targets"].size(-1)
    )
    loss_mask = torch.zeros(
        len(known_mels), max_len, dtype=torch.bool, device=rollout.device
    )
    for row, mel in enumerate(known_mels):
        rollout[row, : mel.size(0)] = mel
        prompt_len = int(prompt_steps[row].item())
        target_len = int(target_lengths[row].item())
        loss_mask[row, prompt_len : min(mel.size(0), target_len)] = True
    return rollout, loss_mask


def masked_refinement_loss(refined, targets, mask):
    expanded_mask = mask.unsqueeze(-1).expand_as(refined)
    difference = refined.float() - targets[:, : refined.size(1)].float()
    denominator = expanded_mask.sum().clamp_min(1)
    l1 = (difference.abs() * expanded_mask).sum() / denominator
    l2 = (difference.square() * expanded_mask).sum() / denominator
    return l1 + l2, l1, l2


def save_checkpoint(path, model, optimizer, step, model_args, mel_config, source):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": None,
            "iter_num": step,
            "model_args": asdict(model_args),
            "mel_config": asdict(mel_config),
            "training_stage": "postnet",
            "source_checkpoint": os.path.abspath(source),
        },
        path,
    )


def main():
    args = parse_args()
    if args.rollout_steps < 1:
        raise ValueError("--rollout-steps must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_args = MelleModelArgs(**checkpoint["model_args"])
    mel_config = MelConfig(**checkpoint["mel_config"])
    model = MelleModel(model_args)
    model.load_state_dict(checkpoint["model"])
    model.to(device=device, dtype=dtype)

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.postnet.parameters():
        parameter.requires_grad_(True)
    model.eval()
    model.postnet.train()

    optimizer = torch.optim.AdamW(
        model.postnet.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        fused=device == "cuda",
    )
    start_iter = 0
    if checkpoint.get("training_stage") == "postnet" and checkpoint.get("optimizer"):
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_iter = int(checkpoint.get("iter_num", -1)) + 1

    dataset = MelleDataset(
        args.manifest,
        tokenizer_path=args.tokenizer,
        mel_config=mel_config,
        max_seq_len=model_args.max_seq_len,
        max_duration_sec=10.0,
        min_duration_sec=6.0,
    )
    if len(dataset) < args.batch_size:
        raise ValueError(
            f"dataset has {len(dataset)} samples, fewer than batch size {args.batch_size}"
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=melle_collate_fn,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    iterator = cycle(loader)

    output_dir = args.output_dir or (
        "runs/melle_postnet_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    )
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(output_dir)
    autocast = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device == "cuda"
        else nullcontext()
    )

    progress = tqdm(
        range(start_iter, args.max_iters + 1),
        total=args.max_iters + 1,
        initial=start_iter,
        desc="MELLE post-net",
        dynamic_ncols=True,
    )
    for step in progress:
        batch = move_batch(next(iterator), device)
        rollout, loss_mask = autoregressive_rollout(
            model, batch, args.rollout_steps, args.sample_latent
        )
        if rollout is None:
            continue

        optimizer.zero_grad(set_to_none=True)
        with autocast:
            refined = model.refine_mel(rollout.detach().to(dtype))
            loss, l1, l2 = masked_refinement_loss(
                refined, batch["mel_targets"], loss_mask
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.postnet.parameters(), 1.0)
        optimizer.step()

        if step % args.log_interval == 0:
            values = {"loss": loss.item(), "l1": l1.item(), "l2": l2.item()}
            progress.set_postfix(**{key: f"{value:.4f}" for key, value in values.items()})
            for key, value in values.items():
                writer.add_scalar(f"postnet/{key}", value, step)

        if step % args.save_interval == 0 or step == args.max_iters:
            save_checkpoint(
                os.path.join(checkpoint_dir, "ckpt.pt"),
                model,
                optimizer,
                step,
                model_args,
                mel_config,
                args.checkpoint,
            )

    writer.close()


if __name__ == "__main__":
    main()
