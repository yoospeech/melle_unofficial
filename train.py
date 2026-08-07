import os
import time
import math
from datetime import datetime
from dataclasses import asdict
from contextlib import nullcontext

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader, DistributedSampler, random_split
from torch.utils.tensorboard import SummaryWriter

from gpt_decoder import Transformer, ModelArgs
from dataset import SpeechToSpeechDataset, pre_training_collate_fn

# -----------------------------------------------------------------------------
# Hyperparameters & Config
# -----------------------------------------------------------------------------
# I/O

LOG_INTERVAL = 100
EVAL_INTERVAL = 500
EVAL_ITERS = 50
SAVE_CKPT = True
RESUME_CKPT = os.environ.get("RESUME_CKPT", "")

# Data
MANIFEST_PATH = os.environ.get("MANIFEST_PATH", "manifest.json")
BATCH_SIZE = 16
MAX_SEQ_LEN = 2048
MAX_DURATION_SEC = 10.0
MIN_DURATION_SEC = 4.0

# Model
DIM = 512
N_LAYERS = 12
N_HEADS = 8
DROPOUT = 0.1

# Optimizer
LEARNING_RATE = 5e-5
MAX_ITERS = 400000
WEIGHT_DECAY = 1e-1
WARMUP_ITERS = 20000
MIN_LR = 5e-7
GRAD_CLIP = 1.0

# System
DEVICE = "cuda" # 'cuda', 'cpu', 'mps'
DTYPE = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
COMPILE = False

# Logging
USE_TENSORBOARD = True
LOG_DIR = "runs/" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
OUT_DIR = os.path.join(LOG_DIR, "checkpoints")

# -----------------------------------------------------------------------------
# Setup DDP & Device
# -----------------------------------------------------------------------------
ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    device = DEVICE
    if torch.backends.mps.is_available():
        device = "mps"
        DTYPE = "float32"
        COMPILE = False

torch.manual_seed(1337 + seed_offset)
if master_process:
    os.makedirs(OUT_DIR, exist_ok=True)

# Mixed Precision Setup
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[DTYPE]
ctx = nullcontext() if device in ["cpu", "mps"] else torch.amp.autocast(device_type="cuda", dtype=ptdtype)
scaler = torch.cuda.amp.GradScaler(enabled=(DTYPE == "float16"))

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------
full_dataset = SpeechToSpeechDataset(
    MANIFEST_PATH,
    max_seq_len=MAX_SEQ_LEN,
    max_duration_sec=MAX_DURATION_SEC,
    min_duration_sec=MIN_DURATION_SEC
)
if len(full_dataset) == 0:
    raise ValueError(
        f"No samples after duration filter (max_duration_sec={MAX_DURATION_SEC}). "
        "Lower the threshold or check manifest duration values."
    )
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Calculate Total Vocab Size dynamically
AUDIO_VOCAB_SIZE = full_dataset.model.feature_extractor.encodec.quantizer.bins
TEXT_VOCAB_SIZE = full_dataset.tokenizer.vocab_size
TOTAL_VOCAB_SIZE = AUDIO_VOCAB_SIZE + TEXT_VOCAB_SIZE

if master_process:
    print(f"Vocab Size Config:")
    print(f"  Audio Codes:      {AUDIO_VOCAB_SIZE}")
    print(f"  Text Tokens:      {TEXT_VOCAB_SIZE}")
    print(f"  Total Vocab Size: {TOTAL_VOCAB_SIZE}")

def get_loader(dataset, shuffle=True):
    sampler = DistributedSampler(dataset) if ddp else None
    return DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=(shuffle and sampler is None), 
        sampler=sampler, 
        collate_fn=pre_training_collate_fn,
        num_workers=0,
        pin_memory=True
    )

train_loader = get_loader(train_dataset, shuffle=True)
val_loader = get_loader(val_dataset, shuffle=False)

# Infinite iterator helper
def cycle(loader):
    while True:
        if ddp and hasattr(loader.sampler, 'set_epoch'):
            loader.sampler.set_epoch(0) # Simplified
        for input_ids, labels in loader:
            yield (
                input_ids.to(device, non_blocking=True),
                labels.to(device, non_blocking=True)
            )

train_iter = cycle(train_loader)
val_iter = cycle(val_loader)

# -----------------------------------------------------------------------------
# Model & Optimizer
# -----------------------------------------------------------------------------
model_args = ModelArgs(
    dim=DIM, n_layers=N_LAYERS, n_heads=N_HEADS, vocab_size=TOTAL_VOCAB_SIZE,
    max_seq_len=MAX_SEQ_LEN, dropout=DROPOUT
)
model = Transformer(model_args).to(device)

print(model)

if COMPILE:
    print("Compiling model...")
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

optimizer = model.configure_optimizers(WEIGHT_DECAY, LEARNING_RATE, (0.9, 0.95), device)
raw_model = model.module if ddp else model

start_iter = 0
if RESUME_CKPT:
    if master_process:
        print(f"Resuming from checkpoint: {RESUME_CKPT}")
    checkpoint = torch.load(RESUME_CKPT, map_location=device, weights_only=False)
    state_dict = checkpoint["model"]
    # Remove _orig_mod. or module. prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[len("_orig_mod."):]] = v
        elif k.startswith("module."):
            new_state_dict[k[len("module."):]] = v
        else:
            new_state_dict[k] = v
    raw_model.load_state_dict(new_state_dict)
    optimizer.load_state_dict(checkpoint["optimizer"])
    if "scaler" in checkpoint and checkpoint["scaler"] is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    start_iter = int(checkpoint.get("iter_num", -1)) + 1
    if master_process:
        print(f"Resumed at iteration {start_iter}")

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def get_lr(it):
    if it < WARMUP_ITERS:
        return LEARNING_RATE * it / WARMUP_ITERS
    if it > MAX_ITERS:
        return MIN_LR
    decay_ratio = (it - WARMUP_ITERS) / (MAX_ITERS - WARMUP_ITERS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)


def token_accuracy_topk(logits, labels, k=10, ignore_index=-1):
    k = min(k, logits.size(-1))
    topk = logits.topk(k, dim=-1).indices
    mask = labels != ignore_index
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    hits = topk.eq(labels.unsqueeze(-1)).any(dim=-1) & mask
    return hits.sum().float() / mask.sum().float()

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split, iterator in [("train", train_iter), ("val", val_iter)]:
        losses = torch.zeros(EVAL_ITERS)
        top10_accs = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            input_ids, labels = next(iterator)
            with ctx:
                logits= model(input_ids, targets=labels)
                loss = raw_model.last_loss
                top10_acc = token_accuracy_topk(logits, labels, k=10)
            losses[k] = loss.item()
            top10_accs[k] = top10_acc.item()
        out[split] = losses.mean()
        out[f"{split}_top10_acc"] = top10_accs.mean().item()
    model.train()
    return out


# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------
if master_process and USE_TENSORBOARD:
    writer = SummaryWriter(log_dir=LOG_DIR)

print(f"Starting training on {device}...")
input_ids, labels = next(train_iter)
t0 = time.time()

for iter_num in range(start_iter, MAX_ITERS + 1):
    # LR Scheduling
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # Evaluation
    if iter_num % EVAL_INTERVAL == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
              f", train top10 acc {losses['train_top10_acc']:.4f}, val top10 acc {losses['val_top10_acc']:.4f}")
        
        if USE_TENSORBOARD:
            writer.add_scalar("loss/train", losses['train'], iter_num)
            writer.add_scalar("loss/val", losses['val'], iter_num)
            writer.add_scalar("lr", lr, iter_num)
            writer.add_scalar("acc_top10/train", losses['train_top10_acc'], iter_num)
            writer.add_scalar("acc_top10/val", losses['val_top10_acc'], iter_num)
            
        if SAVE_CKPT:
            ckpt = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "iter_num": iter_num,
                "config": asdict(model_args),
            }
            torch.save(ckpt, os.path.join(OUT_DIR, "ckpt.pt"))

    # Training Step
    with ctx:
        model(input_ids, targets=labels)
        loss = raw_model.last_loss
        # Optional: compute and log top-k accuracy
    
    input_ids, labels = next(train_iter) # Prefetch next batch
    
    scaler.scale(loss).backward()
    if GRAD_CLIP != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # Logging
    if iter_num % LOG_INTERVAL == 0 and master_process:
        t1 = time.time()
        dt = (t1 - t0) * 1000
        t0 = t1
        print(f"{iter_num} | loss {loss.item():.4f} | lr {lr:.2e} | {dt:.2f}ms")

if ddp:
    destroy_process_group()
if master_process and USE_TENSORBOARD:
    writer.close()
