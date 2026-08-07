"""Losses from the MELLE paper (arXiv:2407.08551v2)."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    expanded = mask.expand_as(values)
    return (values * expanded).sum() / expanded.sum().clamp_min(1.0)


def melle_loss(
    outputs: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    mel_mask: torch.Tensor,
    loss_mask: torch.Tensor,
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

    regression = (
        _masked_mean((coarse - targets).abs(), loss_mask)
        + _masked_mean((coarse - targets).square(), loss_mask)
        + _masked_mean((refined - targets).abs(), loss_mask)
        + _masked_mean((refined - targets).square(), loss_mask)
    )
    kl_values = 0.5 * (logvar.exp() + (mu - targets).square() - 1.0 - logvar)
    kl = _masked_mean(kl_values, loss_mask)

    # The first continuation frame is compared with the final prompt frame.
    flux_mask = loss_mask[:, 1:] & mel_mask[:, :-1]
    if targets.size(1) > 1 and flux_mask.any():
        flux = -_masked_mean((mu[:, 1:] - targets[:, :-1]).abs(), flux_mask)
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
    stop = _masked_mean(stop_elementwise, loss_mask)
    total = regression + kl_weight * kl + flux_weight * flux + stop_weight * stop
    return {
        "loss": total,
        "regression": regression,
        "kl": kl,
        "flux": flux,
        "stop": stop,
    }
