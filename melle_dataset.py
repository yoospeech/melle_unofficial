"""Dataset and collation utilities for continuous-token MELLE training."""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import librosa
import torch
import torchaudio
from torch.utils.data import Dataset

from melle_tokenizer import MelleBPETokenizer

# The repository vendors Vocos as a standalone Python project under ./vocos.
# Put that project root first so its canonical ``vocos.*`` imports resolve in
# training scripts and DataLoader worker processes without an editable install.
VOCOS_PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vocos")
if VOCOS_PROJECT_ROOT not in sys.path:
    sys.path.insert(0, VOCOS_PROJECT_ROOT)

from vocos.feature_extractors import MelSpectrogramFeatures


@dataclass(frozen=True)
class MelConfig:
    sample_rate: int = 24000
    n_fft: int = 1024
    win_length: int = 1024
    hop_length: int = 256
    n_mels: int = 100
    padding: str = "center"
    reduction_factor: int = 1
    prompt_duration_sec: float = 3.0

    @property
    def feature_dim(self) -> int:
        return self.n_mels * self.reduction_factor

    @property
    def prompt_frames(self) -> int:
        frames_per_second = self.sample_rate / self.hop_length
        return round(self.prompt_duration_sec * frames_per_second)

    @property
    def prompt_steps(self) -> int:
        return math.ceil(self.prompt_frames / self.reduction_factor)


class MelleDataset(Dataset):
    """Return text tokens and continuous log-mel targets.

    The manifest contract is intentionally the same as ``dataset.py``:
    ``audio_filepath``, ``text`` and optionally ``duration``.
    """

    def __init__(
        self,
        manifest_path: str,
        tokenizer_path: str = "melle_tokenizer.model",
        mel_config: MelConfig = MelConfig(),
        max_seq_len: int = 2048,
        max_duration_sec: Optional[float] = 10.0,
        min_duration_sec: float = 4.0,
    ) -> None:
        if mel_config.reduction_factor < 1:
            raise ValueError("reduction_factor must be at least 1")
        self.mel_config = mel_config
        self.max_seq_len = max_seq_len
        self.feature_extractor = MelSpectrogramFeatures(
            sample_rate=mel_config.sample_rate,
            n_fft=mel_config.n_fft,
            hop_length=mel_config.hop_length,
            n_mels=mel_config.n_mels,
            padding=mel_config.padding,
        ).eval()

        with open(manifest_path, "r", encoding="utf-8") as handle:
            records = json.load(handle)
        if not isinstance(records, list):
            raise ValueError("manifest must contain a JSON array")
        minimum_continuation_duration = (
            mel_config.prompt_duration_sec
            + mel_config.hop_length / mel_config.sample_rate
        )
        effective_min_duration = max(min_duration_sec, minimum_continuation_duration)
        self.data = self._filter_records(
            records,
            min_duration_sec=effective_min_duration,
            max_duration_sec=max_duration_sec,
        )

        self.tokenizer = MelleBPETokenizer(tokenizer_path, vocab_size=4000)
        self.tokenizer.load_or_train(manifest_path)

    @staticmethod
    def _filter_records(records, min_duration_sec, max_duration_sec):
        filtered = []
        for item in records:
            audio_path = item.get("audio_filepath")
            if not audio_path or "text" not in item:
                continue
            duration = item.get("duration")
            if duration is None:
                try:
                    duration = librosa.get_duration(path=audio_path)
                except Exception:
                    continue
            if duration < min_duration_sec:
                continue
            if max_duration_sec is not None and duration > max_duration_sec:
                continue
            normalized = dict(item)
            normalized["duration"] = float(duration)
            filtered.append(normalized)
        return filtered

    def __len__(self) -> int:
        return len(self.data)

    def _extract_log_mel(self, audio_path: str) -> torch.Tensor:
        cfg = self.mel_config
        waveform, source_rate = torchaudio.load(audio_path)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if source_rate != cfg.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=source_rate, new_freq=cfg.sample_rate
            )
        with torch.no_grad():
            # Vocos returns [B, mel, frames]; MELLE uses [frames, mel].
            features = self.feature_extractor(waveform).squeeze(0).transpose(0, 1)
        return features.contiguous().float()

    def _group_frames(self, mel: torch.Tensor) -> torch.Tensor:
        r = self.mel_config.reduction_factor
        if r == 1:
            return mel
        pad_frames = (-mel.size(0)) % r
        if pad_frames:
            mel = torch.cat([mel, mel[-1:].expand(pad_frames, -1)], dim=0)
        return mel.reshape(mel.size(0) // r, -1)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = self.data[index]
        text_ids: List[int] = [self.tokenizer.bos_token_id]
        text_ids.extend(self.tokenizer.encode(item["text"]))
        text_ids.append(self.tokenizer.eos_token_id)
        text = torch.tensor(text_ids, dtype=torch.long)

        mel = self._group_frames(self._extract_log_mel(item["audio_filepath"]))
        max_mel_len = self.max_seq_len - text.numel()
        if max_mel_len < 1:
            raise ValueError(
                f"text sequence ({text.numel()}) leaves no room for mel frames "
                f"within max_seq_len={self.max_seq_len}"
            )
        mel = mel[:max_mel_len]
        prompt_length = self.mel_config.prompt_steps
        if mel.size(0) <= prompt_length:
            raise ValueError(
                f"audio must contain more than the fixed "
                f"{self.mel_config.prompt_duration_sec:.1f}s prompt: "
                f"{item['audio_filepath']} produced {mel.size(0)} mel frames, "
                f"but prompt requires {prompt_length}"
            )
        return {
            "text_ids": text,
            "mel_targets": mel,
            "prompt_length": torch.tensor(prompt_length, dtype=torch.long),
        }


def melle_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not batch:
        raise ValueError("cannot collate an empty batch")
    batch_size = len(batch)
    max_text = max(item["text_ids"].numel() for item in batch)
    max_mel = max(item["mel_targets"].size(0) for item in batch)
    feature_dim = batch[0]["mel_targets"].size(1)

    text_ids = torch.zeros(batch_size, max_text, dtype=torch.long)
    text_mask = torch.zeros(batch_size, max_text, dtype=torch.bool)
    mel_inputs = torch.zeros(batch_size, max_mel, feature_dim)
    mel_targets = torch.zeros_like(mel_inputs)
    mel_mask = torch.zeros(batch_size, max_mel, dtype=torch.bool)
    loss_mask = torch.zeros(batch_size, max_mel, dtype=torch.bool)
    stop_targets = torch.zeros(batch_size, max_mel)

    for row, item in enumerate(batch):
        text = item["text_ids"]
        mel = item["mel_targets"]
        if mel.size(1) != feature_dim:
            raise ValueError("all mel targets must have the same feature dimension")
        text_len, mel_len = text.numel(), mel.size(0)
        prompt_len = int(item["prompt_length"].item())
        text_ids[row, :text_len] = text
        text_mask[row, :text_len] = True
        mel_targets[row, :mel_len] = mel
        mel_mask[row, :mel_len] = True
        loss_mask[row, prompt_len:mel_len] = True
        if mel_len > 1:
            mel_inputs[row, 1:mel_len] = mel[:-1]
        stop_targets[row, mel_len - 1] = 1.0

    return {
        "text_ids": text_ids,
        "text_mask": text_mask,
        "mel_inputs": mel_inputs,
        "mel_targets": mel_targets,
        "mel_mask": mel_mask,
        "loss_mask": loss_mask,
        "prompt_lengths": torch.tensor(
            [int(item["prompt_length"].item()) for item in batch], dtype=torch.long
        ),
        "stop_targets": stop_targets,
    }
