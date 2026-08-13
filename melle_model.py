"""Continuous mel-spectrogram autoregressive model inspired by MELLE."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from gpt_decoder import ModelArgs, RMSNorm, TransformerBlock, precompute_freqs_cis


@dataclass
class MelleModelArgs:
    text_vocab_size: int
    mel_dim: int = 100
    dim: int = 1024
    n_layers: int = 12
    n_heads: int = 16
    hidden_dim: int | None = 4096
    multiple_of: int = 256
    max_seq_len: int = 2048
    dropout: float = 0.1
    prenet_dropout: float = 0.5
    postnet_channels: int = 256
    postnet_layers: int = 5
    postnet_kernel_size: int = 5
    norm_eps: float = 1e-5


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MelPrenet(nn.Module):
    """Three-layer mel projection with inference-time dropout.

    MELLE follows Tacotron's use of pre-net dropout as a source of variation
    during both training and inference.
    """

    def __init__(self, mel_dim: int, model_dim: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(mel_dim, model_dim),
                nn.Linear(model_dim, model_dim),
                nn.Linear(model_dim, model_dim),
            ]
        )
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=True)
        return x


class PostNet(nn.Module):
    def __init__(self, mel_dim: int, channels: int, layers: int, kernel_size: int, dropout: float):
        super().__init__()
        if layers < 2:
            raise ValueError("postnet_layers must be at least 2")
        padding = (kernel_size - 1) // 2
        blocks = []
        in_channels = mel_dim
        for _ in range(layers - 1):
            blocks.extend(
                [
                    nn.Conv1d(
                        in_channels, channels, kernel_size, padding=padding
                    ),
                    nn.Tanh(),
                    nn.Dropout(dropout),
                ]
            )
            in_channels = channels
        blocks.append(nn.Conv1d(in_channels, mel_dim, kernel_size, padding=padding))
        self.net = nn.Sequential(*blocks)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.net(mel.transpose(1, 2)).transpose(1, 2)


class MelleModel(nn.Module):
    def __init__(self, args: MelleModelArgs):
        super().__init__()
        if args.dim % args.n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        if args.postnet_kernel_size % 2 == 0:
            raise ValueError("postnet_kernel_size must be odd")
        self.args = args
        self.text_embedding = nn.Embedding(args.text_vocab_size, args.dim)
        self.mel_prenet = MelPrenet(args.mel_dim, args.dim, args.prenet_dropout)
        transformer_args = ModelArgs(
            dim=args.dim,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            vocab_size=args.text_vocab_size,
            hidden_dim=args.hidden_dim,
            multiple_of=args.multiple_of,
            norm_eps=args.norm_eps,
            max_seq_len=args.max_seq_len,
            dropout=args.dropout,
        )
        self.layers = nn.ModuleList(
            TransformerBlock(layer_id, transformer_args) for layer_id in range(args.n_layers)
        )
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.latent_stats = nn.Linear(args.dim, 2 * args.mel_dim)
        self.latent_decoder = MLP(args.mel_dim, args.dim, args.mel_dim, args.dropout)
        self.stop_head = nn.Linear(args.dim, 1)
        self.postnet = PostNet(
            args.mel_dim,
            args.postnet_channels,
            args.postnet_layers,
            args.postnet_kernel_size,
            args.dropout,
        )
        freqs_cos, freqs_sin = precompute_freqs_cis(
            args.dim // args.n_heads, args.max_seq_len
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
        self.apply(self._init_weights)
        for name, parameter in self.named_parameters():
            if name.endswith("w3.weight") or name.endswith("wo.weight"):
                nn.init.normal_(parameter, mean=0.0, std=0.02 / math.sqrt(2 * args.n_layers))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _pack_inputs(
        self,
        text_embeddings: torch.Tensor,
        text_mask: torch.Tensor,
        mel_embeddings: torch.Tensor,
        mel_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = text_embeddings.size(0)
        text_lengths = text_mask.sum(dim=1)
        mel_lengths = mel_mask.sum(dim=1)
        total_lengths = text_lengths + mel_lengths
        max_total = int(total_lengths.max().item())
        if max_total > self.args.max_seq_len:
            raise ValueError(
                f"packed sequence length {max_total} exceeds max_seq_len={self.args.max_seq_len}"
            )
        packed = text_embeddings.new_zeros(batch_size, max_total, self.args.dim)
        # Acoustic predictions are aligned as
        # [EOS hidden, y[0] hidden, ..., y[T-2] hidden] -> [y[0], ..., y[T-1]].
        acoustic_positions = torch.zeros(
            batch_size,
            mel_embeddings.size(1) + 1,
            dtype=torch.long,
            device=packed.device,
        )
        text_rows, text_columns = text_mask.nonzero(as_tuple=True)
        packed[text_rows, text_columns] = text_embeddings[text_rows, text_columns]

        mel_rows, mel_columns = mel_mask.nonzero(as_tuple=True)
        mel_destinations = text_lengths[mel_rows] + mel_columns
        packed[mel_rows, mel_destinations] = mel_embeddings[mel_rows, mel_columns]
        eos_positions = text_lengths - 1
        acoustic_positions[:, 0] = eos_positions
        if mel_embeddings.size(1) > 0:
            all_mel_positions = text_lengths.unsqueeze(1) + torch.arange(
                mel_embeddings.size(1), device=packed.device
            ).unsqueeze(0)
            # Padded positions are masked from every loss, but gather still
            # requires placeholder indices to lie inside the packed tensor.
            acoustic_positions[:, 1:] = all_mel_positions.clamp_max(max_total - 1)
        acoustic_lengths = mel_lengths + 1
        acoustic_mask = torch.arange(
            mel_embeddings.size(1) + 1, device=packed.device
        ).unsqueeze(0) < acoustic_lengths.unsqueeze(1)
        return packed, acoustic_positions, acoustic_mask

    @staticmethod
    def _text_prefix_attention_mask(
        text_lengths: torch.Tensor,
        mel_lengths: torch.Tensor,
        sequence_length: int,
    ) -> torch.Tensor:
        """Allow bidirectional text attention and causal attention over all mel.

        The factorization is ``p(y_t | x, y_<t)``: the complete text ``x`` is
        visible as a fixed prefix, while all mel frames belong to the
        autoregressive ``y`` sequence. Acoustic prompting is an inference
        operation; training predicts the complete mel target.
        """
        total_lengths = text_lengths + mel_lengths
        positions = torch.arange(sequence_length, device=text_lengths.device)
        queries = positions.view(1, sequence_length, 1)
        keys = positions.view(1, 1, sequence_length)
        valid_queries = queries < total_lengths.view(-1, 1, 1)
        valid_keys = keys < total_lengths.view(-1, 1, 1)
        prefix = text_lengths.view(-1, 1, 1)
        prefix_to_prefix = (queries < prefix) & (keys < prefix)
        mel_causal = (queries >= prefix) & (keys <= queries)
        allowed = valid_queries & valid_keys & (prefix_to_prefix | mel_causal)
        return allowed.unsqueeze(1)

    def forward(
        self,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
        mel_inputs: torch.Tensor,
        mel_mask: torch.Tensor,
        sample_latent: bool = True,
        apply_postnet: bool = True,
    ) -> Dict[str, torch.Tensor]:
        text_embeddings = self.text_embedding(text_ids)
        mel_embeddings = self.mel_prenet(mel_inputs)
        # Autocast can leave nn.Embedding in FP32 while the mel MLP runs in
        # BF16/FP16. Packing uses indexed assignment, which requires an exact
        # dtype match between both modality embeddings.
        text_embeddings = text_embeddings.to(dtype=mel_embeddings.dtype)
        hidden, acoustic_positions, acoustic_mask = self._pack_inputs(
            text_embeddings, text_mask, mel_embeddings, mel_mask
        )
        seq_len = hidden.size(1)
        attention_mask = self._text_prefix_attention_mask(
            text_mask.sum(dim=1),
            mel_mask.sum(dim=1),
            seq_len,
        )
        for layer in self.layers:
            hidden = layer(
                hidden,
                self.freqs_cos[:seq_len],
                self.freqs_sin[:seq_len],
                attention_mask,
            )
        hidden = self.norm(hidden)
        mel_hidden = hidden.gather(
            1, acoustic_positions.unsqueeze(-1).expand(-1, -1, hidden.size(-1))
        )

        mu, logvar = self.latent_stats(mel_hidden).chunk(2, dim=-1)
        logvar = logvar.clamp(min=-10.0, max=10.0)
        if sample_latent:
            latent = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        else:
            latent = mu
        coarse = latent + self.latent_decoder(latent)
        # Values gathered for padded mel positions are placeholders. Zero them
        # before the convolutional post-net so they cannot leak into the final
        # valid frames through the convolution kernel.
        coarse = coarse * acoustic_mask.unsqueeze(-1).to(coarse.dtype)
        refined = self.refine_mel(coarse) if apply_postnet else coarse
        refined = refined * acoustic_mask.unsqueeze(-1).to(refined.dtype)
        return {
            "coarse_mel": coarse,
            "refined_mel": refined,
            "mu": mu,
            "logvar": logvar,
            "stop_logits": self.stop_head(mel_hidden).squeeze(-1),
        }

    def refine_mel(self, coarse: torch.Tensor) -> torch.Tensor:
        """Refine a completed coarse mel sequence with the non-causal post-net."""
        return coarse + self.postnet(coarse)
