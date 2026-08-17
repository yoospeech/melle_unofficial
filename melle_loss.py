"""Losses from the MELLE paper (arXiv:2407.08551v2)."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def _masked_sum(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    expanded = mask.expand_as(values)
    return (values * expanded).sum()


def melle_loss(
    outputs: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    mel_mask: torch.Tensor,
    stop_targets: torch.Tensor,
    kl_weight: float,
    flux_weight: float = 0.5,
    stop_weight: float = 1.0,
    stop_positive_weight: float = 100.0,
) -> Dict[str, torch.Tensor]:
    coarse = outputs["coarse_mel"]
    refined = outputs["refined_mel"]
    mu = outputs["mu"]
    logvar = outputs["logvar"]
    coarse_regression = _masked_sum((coarse - targets).abs(), mel_mask) + _masked_sum(
        (coarse - targets).square(), mel_mask
    )
    refined_regression = _masked_sum(
        (refined - targets).abs(), mel_mask
    ) + _masked_sum((refined - targets).square(), mel_mask)
    # Both heads are supervised in one graph. The coarse term prevents the
    # autoregressive output from delegating all reconstruction to the post-net.
    regression = coarse_regression + refined_regression
    kl_values = 0.5 * (logvar.exp() + (mu - targets).square() - 1.0 - logvar)
    kl = _masked_sum(kl_values, mel_mask)

    # Equation (9) of MELLE rewards variation from the preceding ground-truth
    # frame: L_flux = -sum_t ||mu_t - y_{t-1}||_1. Keep the time dimension
    # intact so the final frame of one utterance is never paired with the first
    # frame of another utterance after batching.
    flux_mask = mel_mask[:, 1:] & mel_mask[:, :-1]
    if targets.size(1) > 1 and flux_mask.any():
        flux = -_masked_sum((mu[:, 1:] - targets[:, :-1]).abs(), flux_mask)
    else:
        flux = mu.sum() * 0.0

    stop_elementwise = F.binary_cross_entropy_with_logits(
        outputs["stop_logits"],
        stop_targets,
        reduction="none",
        pos_weight=torch.as_tensor(
            stop_positive_weight,
            device=stop_targets.device,
            dtype=stop_targets.dtype,
        ),
    )
    # Stop prediction is defined at every real acoustic position, including
    # prompt positions. Only padded positions are excluded.
    stop = _masked_sum(stop_elementwise, mel_mask)

    # Match melle_comp's reported unit: sum over mel channels, then average all
    # objectives by the number of valid acoustic frames in the padded batch.
    # mel_mask is the target-frame mask supplied by the collate function, so
    # no decoder-side mask needs to be returned for this normalization.
    valid_frames = mel_mask.sum().to(dtype=targets.dtype).clamp_min(1.0)
    coarse_regression = coarse_regression / valid_frames
    refined_regression = refined_regression / valid_frames
    regression = regression / valid_frames
    kl = kl / valid_frames
    flux = flux / valid_frames
    stop = stop / valid_frames

    total = regression + kl_weight * kl + flux_weight * flux + stop_weight * stop
    return {
        "loss": total,
        "regression": regression,
        "coarse_regression": coarse_regression,
        "refined_regression": refined_regression,
        "kl": kl,
        "flux": flux,
        "stop": stop,
    }
