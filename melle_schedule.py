"""Training schedules used by the MELLE paper configuration."""

from __future__ import annotations


def linear_warmup_decay_lr(
    step: int,
    *,
    peak_lr: float,
    warmup_steps: int,
    total_steps: int,
    min_lr: float = 0.0,
) -> float:
    """Warm up linearly to ``peak_lr``, then decay linearly to ``min_lr``.

    ``step`` is the zero-based optimizer-update index. Consequently, step 0
    uses zero learning rate, ``warmup_steps`` reaches the peak, and
    ``total_steps`` reaches the minimum. Values beyond the configured training
    interval are clamped to the nearest endpoint, which also makes checkpoint
    resume deterministic.
    """
    if step < 0:
        raise ValueError("step must be non-negative")
    if peak_lr < 0.0 or min_lr < 0.0:
        raise ValueError("learning rates must be non-negative")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if total_steps <= warmup_steps:
        raise ValueError("total_steps must be greater than warmup_steps")

    if warmup_steps and step < warmup_steps:
        return peak_lr * step / warmup_steps

    decay_progress = (step - warmup_steps) / (total_steps - warmup_steps)
    decay_progress = min(1.0, max(0.0, decay_progress))
    return peak_lr + decay_progress * (min_lr - peak_lr)


def delayed_weight(step: int, *, weight: float, delay_steps: int) -> float:
    """Return zero during a delay and the full weight from its boundary on."""
    if step < 0:
        raise ValueError("step must be non-negative")
    if weight < 0.0:
        raise ValueError("weight must be non-negative")
    if delay_steps < 0:
        raise ValueError("delay_steps must be non-negative")
    return 0.0 if step < delay_steps else weight
