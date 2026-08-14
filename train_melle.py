"""Train MELLE while retaining the repository's existing training conventions."""

from __future__ import annotations

import os
import time
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime

import torch
import torch.distributed as dist
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from melle_dataset import MelConfig, MelleDataset, melle_collate_fn
from melle_loss import melle_loss
from melle_model import MelleModel, MelleModelArgs
from melle_schedule import delayed_weight, linear_warmup_decay_lr
from melle_tokenizer import MelleCharacterTokenizer


MANIFEST_PATH = os.environ.get("MANIFEST_PATH", "manifest.json")
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "melle_character_tokenizer.model")
RESUME_CKPT = os.environ.get("RESUME_CKPT", "")
LOG_INTERVAL = 5
EVAL_INTERVAL = 500
EVAL_ITERS = 50
SAVE_CKPT = True
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
NUM_WORKERS = min(8, os.cpu_count() or 1)
MAX_SEQ_LEN = 2048
MAX_DURATION_SEC = 10.0
MIN_DURATION_SEC = 4.0
VAL_RATIO = 0.001
REDUCTION_FACTOR = 1
DIM = 1024
N_LAYERS = 12
N_HEADS = 16
HIDDEN_DIM = 4096
DROPOUT = 0.1
LEARNING_RATE = 5e-5
MAX_ITERS = 400_000
WEIGHT_DECAY = 0.1
WARMUP_ITERS = 32_000
KL_WARMUP_ITERS = 10_000
MAX_KL_WEIGHT = 0.1
FLUX_WEIGHT = 0.5
STOP_WEIGHT = 1.0
MIN_LR = 0.0
GRAD_CLIP = 1.0
DEVICE = "cuda"
DTYPE = "bfloat16"
USE_TENSORBOARD = True
LOG_DIR = "runs/melle_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
OUT_DIR = os.path.join(LOG_DIR, "checkpoints")


ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(device)
    master_process = rank == 0
    seed_offset = rank
else:
    world_size = 1
    master_process = True
    seed_offset = 0
    device = DEVICE if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        DTYPE = "float32"

torch.manual_seed(1337 + seed_offset)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

GRAD_ACCUM_STEPS = 1
if master_process:
    os.makedirs(OUT_DIR, exist_ok=True)
    print(
        f"Batch configuration: {BATCH_SIZE} samples/GPU x {world_size} rank(s), "
        f"gradient accumulation={GRAD_ACCUM_STEPS}"
    )

ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[DTYPE]
ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type="cuda", dtype=ptdtype)
scaler = torch.cuda.amp.GradScaler(enabled=(device != "cpu" and DTYPE == "float16"))

# Avoid concurrent SentencePiece writes when launching with torchrun.
if ddp and not os.path.exists(TOKENIZER_PATH):
    if master_process:
        MelleCharacterTokenizer(TOKENIZER_PATH, vocab_size=4000).load_or_train(
            MANIFEST_PATH
        )
    dist.barrier()

mel_config = MelConfig(reduction_factor=REDUCTION_FACTOR)
full_dataset = MelleDataset(
    MANIFEST_PATH,
    tokenizer_path=TOKENIZER_PATH,
    mel_config=mel_config,
    max_seq_len=MAX_SEQ_LEN,
    max_duration_sec=MAX_DURATION_SEC,
    min_duration_sec=MIN_DURATION_SEC,
)
if len(full_dataset) < 2:
    raise ValueError("MELLE training requires at least two samples after duration filtering")
if master_process:
    print(
        f"Duration filter: {MIN_DURATION_SEC:.1f}s to {MAX_DURATION_SEC:.1f}s; "
        f"training uses a fixed {mel_config.prompt_duration_sec:.1f}s acoustic prompt "
        "and supervises the continuation"
    )

val_size = max(1, round(len(full_dataset) * VAL_RATIO))
val_size = min(val_size, len(full_dataset) - 1)
train_size = len(full_dataset) - val_size
split_generator = torch.Generator().manual_seed(1337)
train_dataset, val_dataset = random_split(
    full_dataset, [train_size, val_size], generator=split_generator
)
if master_process:
    print(
        f"Dataset split: train={train_size:,}, validation={val_size:,} "
        f"({val_size / len(full_dataset):.4%})"
    )


def get_loader(dataset, shuffle):
    sampler = DistributedSampler(dataset, shuffle=shuffle) if ddp else None
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        drop_last=shuffle,
        collate_fn=melle_collate_fn,
        num_workers=NUM_WORKERS,
        # DGX Spark uses coherent unified memory. Avoid a second pinned host
        # staging pool and let CUDA migrate pages from ordinary CPU tensors.
        pin_memory=False,
        persistent_workers=NUM_WORKERS > 0,
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )


train_loader = get_loader(train_dataset, True)
val_loader = get_loader(val_dataset, False)


def cycle(loader):
    epoch = 0
    while True:
        if ddp and isinstance(loader.sampler, DistributedSampler):
            loader.sampler.set_epoch(epoch)
        yield from loader
        epoch += 1


train_iter = cycle(train_loader)
val_iter = cycle(val_loader)
model_args = MelleModelArgs(
    text_vocab_size=full_dataset.tokenizer.vocab_size,
    mel_dim=mel_config.feature_dim,
    dim=DIM,
    n_layers=N_LAYERS,
    n_heads=N_HEADS,
    hidden_dim=HIDDEN_DIM,
    max_seq_len=MAX_SEQ_LEN,
    dropout=DROPOUT,
)
model = MelleModel(model_args).to(device)
optimizer_kwargs = {
    "lr": LEARNING_RATE,
    "betas": (0.9, 0.95),
    "weight_decay": WEIGHT_DECAY,
}
if device != "cpu":
    optimizer_kwargs["fused"] = True
optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
if ddp:
    model = DDP(model, device_ids=[local_rank])
raw_model = model.module if ddp else model
start_iter = 0

if RESUME_CKPT:
    checkpoint = torch.load(RESUME_CKPT, map_location=device)
    raw_model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    start_iter = int(checkpoint.get("iter_num", -1)) + 1


def move_batch(batch):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def get_lr(step):
    return linear_warmup_decay_lr(
        step,
        peak_lr=LEARNING_RATE,
        warmup_steps=WARMUP_ITERS,
        total_steps=MAX_ITERS,
        min_lr=MIN_LR,
    )


def compute_losses(batch, step, sample_latent=True):
    outputs = model(
        batch["text_ids"],
        batch["text_mask"],
        batch["mel_inputs"],
        batch["mel_input_mask"],
        sample_latent=sample_latent,
    )
    # Section 4.2: disable KL for the first 10K updates, then enable its full
    # weight. Spectrogram flux has no warmup and is active from step zero.
    kl_weight = delayed_weight(
        step, weight=MAX_KL_WEIGHT, delay_steps=KL_WARMUP_ITERS
    )
    return melle_loss(
        outputs,
        batch["mel_targets"],
        batch["mel_target_mask"],
        batch["loss_mask"],
        batch["stop_targets"],
        kl_weight=kl_weight,
        flux_weight=FLUX_WEIGHT,
        stop_weight=STOP_WEIGHT,
    )


@torch.no_grad()
def estimate_loss(step):
    model.eval()
    result = {}
    for split, iterator in (("train", train_iter), ("val", val_iter)):
        totals = {name: 0.0 for name in ("loss", "regression", "kl", "flux", "stop")}
        eval_progress = tqdm(
            range(EVAL_ITERS),
            desc=f"eval/{split}",
            leave=False,
            dynamic_ncols=True,
            disable=not master_process,
        )
        for _ in eval_progress:
            batch = move_batch(next(iterator))
            with ctx:
                losses = compute_losses(batch, step, sample_latent=False)
            for name in totals:
                totals[name] += losses[name].item() / EVAL_ITERS
            eval_progress.set_postfix(loss=f"{losses['loss'].item():.4f}")
        if ddp:
            for name, value in totals.items():
                reduced = torch.tensor(value, device=device)
                dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
                totals[name] = (reduced / world_size).item()
        result[split] = totals
    model.train()
    return result


writer = SummaryWriter(LOG_DIR) if master_process and USE_TENSORBOARD else None
batch = move_batch(next(train_iter))
started = time.time()
last_log_iter = start_iter - 1
optimizer.zero_grad(set_to_none=True)
progress = tqdm(
    range(start_iter, MAX_ITERS + 1),
    total=MAX_ITERS + 1,
    initial=start_iter,
    desc="MELLE",
    dynamic_ncols=True,
    disable=not master_process,
)
for iter_num in progress:
    lr = get_lr(iter_num)
    for group in optimizer.param_groups:
        group["lr"] = lr

    # Every DDP rank participates in evaluation collectives; only rank zero
    # performs user-visible logging and checkpoint writes.
    if iter_num % EVAL_INTERVAL == 0:
        metrics = estimate_loss(iter_num)
        if master_process:
            progress.write(
                f"step {iter_num}: train {metrics['train']['loss']:.4f}, "
                f"val {metrics['val']['loss']:.4f}"
            )
            if writer:
                for split, values in metrics.items():
                    for name, value in values.items():
                        writer.add_scalar(f"{name}/{split}", value, iter_num)
                writer.add_scalar("lr", lr, iter_num)
            if SAVE_CKPT:
                torch.save(
                    {
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(),
                        "iter_num": iter_num,
                        "model_args": asdict(model_args),
                        "mel_config": asdict(mel_config),
                    },
                    os.path.join(OUT_DIR, "ckpt.pt"),
                )

    accumulated = {name: 0.0 for name in ("loss", "regression", "kl", "flux", "stop")}
    for micro_step in range(GRAD_ACCUM_STEPS):
        sync_context = (
            model.no_sync()
            if ddp and micro_step < GRAD_ACCUM_STEPS - 1
            else nullcontext()
        )
        with sync_context:
            with ctx:
                losses = compute_losses(batch, iter_num)
                loss = losses["loss"] / GRAD_ACCUM_STEPS
            scaler.scale(loss).backward()
        for name in accumulated:
            accumulated[name] += losses[name].detach().item() / GRAD_ACCUM_STEPS
        batch = move_batch(next(train_iter))

    if GRAD_CLIP:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    if iter_num % LOG_INTERVAL == 0 and master_process:
        completed_steps = max(1, iter_num - last_log_iter)
        elapsed_ms = (time.time() - started) * 1000 / completed_steps
        started = time.time()
        last_log_iter = iter_num
        progress.set_postfix(
            loss=f"{accumulated['loss']:.4f}",
            reg=f"{accumulated['regression']:.4f}",
            kl=f"{accumulated['kl']:.4f}",
            flux=f"{accumulated['flux']:.4f}",
            stop=f"{accumulated['stop']:.4f}",
            lr=f"{lr:.2e}",
            ms_per_step=f"{elapsed_ms:.1f}",
        )

if ddp:
    destroy_process_group()
if writer:
    writer.close()
