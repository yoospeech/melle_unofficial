import json
import os
import numpy as np
import torch
import librosa
from torch.utils.data import Dataset, DataLoader

from WavTokenizer.decoder.pretrained import WavTokenizer
from WavTokenizer.encoder.utils import convert_audio
from tokenizer import CharacterTokenizer

import random

DEFAULT_WAVTOKENIZER_CONFIG = os.environ.get(
    "WAVTOKENIZER_CONFIG",
    "./WavTokenizer/configs/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
)
DEFAULT_WAVTOKENIZER_CKPT = os.environ.get(
    "WAVTOKENIZER_CKPT",
    "./WavTokenizer/wavtokenizer_large_unify_600_24k.ckpt",
)

class TextSpeechDataset(Dataset):
    """
    Pre-training dataset for:
      speech_in -> text (reasoning / response) -> speech_out

    Manifest format:
    [
      {
        "audio_in": "path/to/input.wav",
        "text": "안녕하세요 반갑습니다. 오늘 기분 좋은 하루 되세요",
        "audio_out": "path/to/output.wav"
      }
    ]
    """

    def __init__(
        self,
        manifest_path,
        tokenizer_path="tokenizer.json",
        max_seq_len=2048,
        config_path=DEFAULT_WAVTOKENIZER_CONFIG,
        model_path=DEFAULT_WAVTOKENIZER_CKPT,
        bandwidth_id=0,
        sample_rate=24000,
        max_duration_sec=10.0,
        min_duration_sec=4.0,
        device="cuda",
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.max_seq_len = max_seq_len
        self.sample_rate = sample_rate
        self.bandwidth_id = torch.tensor([bandwidth_id], device=self.device)

        # ---------- load manifest ----------
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        if max_duration_sec is not None:
            filtered = []
            for item in self.data:
                duration = item.get("duration")
                if duration is None:
                    audio_path = item.get("audio_filepath")
                    if not audio_path:
                        continue
                    try:
                        duration = librosa.get_duration(path=audio_path)
                    except Exception:
                        continue
                if duration <= max_duration_sec and duration >= min_duration_sec:
                    filtered.append(item)
            self.data = filtered

        # ---------- load text tokenizer ----------
        self.tokenizer = CharacterTokenizer()
        if os.path.exists(tokenizer_path):
            self.tokenizer.load(tokenizer_path)
        else:
            print("Training CharacterTokenizer...")
            self.tokenizer.train(manifest_path, save_path=tokenizer_path)

        # ---------- load wavtokenizer ----------
        print("Loading WavTokenizer...")
        self.model = WavTokenizer.from_pretrained0802(
            config_path, model_path
        ).to(self.device)
        self.model.eval()

        
    def __len__(self):
        return len(self.data)


    def _load_or_encode(self, audio_path, top_db=30):

        wav_np, sr = librosa.load(audio_path, sr=None, mono=True)
        wav_np, _ = librosa.effects.trim(wav_np, top_db=top_db)
        wav = torch.from_numpy(wav_np).unsqueeze(0)  # (1, T)
        wav = convert_audio(wav, sr, self.sample_rate, 1).to(self.device)

        with torch.no_grad():
            _, codes = self.model.encode_infer(
                wav, bandwidth_id=self.bandwidth_id
            )
            # [n_q, B, T] -> [n_q, T]
            codes = codes.squeeze(1).cpu().numpy()

        return torch.from_numpy(codes).long()

    def _find_energy_split(self, audio_path, num_frames, min_ratio=0.2, max_ratio=0.8):
        if num_frames <= 1:
            return 1

        try:
            wav_np, _ = librosa.load(audio_path, sr=None, mono=True)
        except Exception:
            return max(1, int(0.3 * num_frames))

        if wav_np.size == 0:
            return max(1, int(0.3 * num_frames))

        frame_size = max(1, len(wav_np) // num_frames)
        energies = np.empty(num_frames, dtype=np.float32)
        for i in range(num_frames):
            start = i * frame_size
            end = len(wav_np) if i == num_frames - 1 else (i + 1) * frame_size
            frame = wav_np[start:end]
            energies[i] = np.mean(frame ** 2) if frame.size > 0 else np.inf

        start_idx = int(min_ratio * num_frames)
        end_idx = int(max_ratio * num_frames)
        if end_idx <= start_idx:
            start_idx = 0
            end_idx = num_frames

        split = start_idx + int(np.argmin(energies[start_idx:end_idx]))
        return max(1, min(split, num_frames - 1))

    # --------------------------------------------------
    # __getitem__
    # --------------------------------------------------

    def __getitem__(self, idx):
        item = self.data[idx]
        audio_path = item["audio_filepath"]
        text = item["text"]
    
        # ---------- audio ----------
        codes = self._load_or_encode(audio_path) + self.tokenizer.vocab_size
        audio_seq = codes[0]  # [T]
    
        T = len(audio_seq)
        #split = self._find_energy_split(audio_path, T)

        split = 120 # hardcoded split for 3s @ 75fps, can be replaced by energy-based split like vall-e
    
        audio_in_seq = audio_seq[:split]
        audio_out_seq = audio_seq[split:] 
    
        # ---------- text ----------
        text_ids = self.tokenizer.encode(text)
        text_ids = (
            [self.tokenizer.bos_token_id]
            + text_ids
            + [self.tokenizer.eos_token_id]
        )
        
        text_ids = torch.tensor(text_ids, dtype=torch.long)
        eoa_token = torch.tensor([self.tokenizer.eoa_token_id], dtype=torch.long)

        # ---------- concat ----------
        full_seq = torch.cat(
            [text_ids, audio_in_seq, audio_out_seq, eoa_token]
        )

        audio_in_len = len(text_ids) + len(audio_in_seq)
    
        return full_seq, audio_in_len


class SpeechTextSpeechDataset(Dataset):
    """
    Pre-training dataset for:
      speech_in -> text (reasoning / response) -> speech_out

    Manifest format:
    [
      {
        "audio_in": "path/to/input.wav",
        "text": "안녕하세요 반갑습니다. 오늘 기분 좋은 하루 되세요",
        "audio_out": "path/to/output.wav"
      }
    ]
    """

    def __init__(
        self,
        manifest_path,
        tokenizer_path="tokenizer.json",
        max_seq_len=2048,
        config_path=DEFAULT_WAVTOKENIZER_CONFIG,
        model_path=DEFAULT_WAVTOKENIZER_CKPT,
        bandwidth_id=0,
        sample_rate=24000,
        max_duration_sec=10.0,
        min_duration_sec=4.0,
        device="cuda",
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.max_seq_len = max_seq_len
        self.sample_rate = sample_rate
        self.bandwidth_id = torch.tensor([bandwidth_id], device=self.device)

        # ---------- load manifest ----------
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        if max_duration_sec is not None:
            filtered = []
            for item in self.data:
                duration = item.get("duration")
                if duration is None:
                    audio_path = item.get("audio_filepath")
                    if not audio_path:
                        continue
                    try:
                        duration = librosa.get_duration(path=audio_path)
                    except Exception:
                        continue
                if duration <= max_duration_sec and duration >= min_duration_sec:
                    filtered.append(item)
            self.data = filtered

        # ---------- load text tokenizer ----------
        self.tokenizer = CharacterTokenizer()
        if os.path.exists(tokenizer_path):
            self.tokenizer.load(tokenizer_path)
        else:
            print("Training CharacterTokenizer...")
            self.tokenizer.train(manifest_path, save_path=tokenizer_path)

        # ---------- load wavtokenizer ----------
        print("Loading WavTokenizer...")
        self.model = WavTokenizer.from_pretrained0802(
            config_path, model_path
        ).to(self.device)
        self.model.eval()

        
    def __len__(self):
        return len(self.data)


    def _load_or_encode(self, audio_path, top_db=30):

        wav_np, sr = librosa.load(audio_path, sr=None, mono=True)
        wav_np, _ = librosa.effects.trim(wav_np, top_db=top_db)
        wav = torch.from_numpy(wav_np).unsqueeze(0)  # (1, T)
        wav = convert_audio(wav, sr, self.sample_rate, 1).to(self.device)

        with torch.no_grad():
            _, codes = self.model.encode_infer(
                wav, bandwidth_id=self.bandwidth_id
            )
            # [n_q, B, T] -> [n_q, T]
            codes = codes.squeeze(1).cpu().numpy()

        return torch.from_numpy(codes).long()

    # --------------------------------------------------
    # __getitem__
    # --------------------------------------------------

    def __getitem__(self, idx):
        item = self.data[idx]
        audio_path = item["audio_filepath"]
        text = item["text"]
    
        # ---------- audio ----------
        codes = self._load_or_encode(audio_path) + self.tokenizer.vocab_size
        audio_seq = codes[0]  # [T]
    
        T = len(audio_seq)
        #split = self._find_energy_split(audio_path, T)

        split = 120 # hardcoded split for 3s @ 40fps, can be replaced by energy-based split like vall-e
    
        audio_in_seq = audio_seq[:split]
        audio_out_seq = audio_seq[split:] 
    
        # ---------- text ----------
        text_ids = self.tokenizer.encode(text)
        text_ids = (
            [self.tokenizer.bos_token_id]
            + text_ids
            + [self.tokenizer.eos_token_id]
        )
        
        text_ids = torch.tensor(text_ids, dtype=torch.long)
        eoa_token = torch.tensor([self.tokenizer.eoa_token_id], dtype=torch.long)

        # ---------- concat ----------
        full_seq = torch.cat(
            [audio_in_seq, text_ids, audio_out_seq, eoa_token]
        )

        audio_in_len = len(audio_in_seq)
    
        return full_seq, audio_in_len



class SpeechToSpeechDataset(Dataset):
    """
    Pre-training dataset for:
      speech_in -> text (reasoning / response) -> speech_out

    Manifest format:
    [
      {
        "audio_in": "path/to/input.wav",
        "text": "안녕하세요 반갑습니다. 오늘 기분 좋은 하루 되세요",
        "audio_out": "path/to/output.wav"
      }
    ]
    """

    def __init__(
        self,
        manifest_path,
        tokenizer_path="tokenizer.json",
        max_seq_len=2048,
        config_path=DEFAULT_WAVTOKENIZER_CONFIG,
        model_path=DEFAULT_WAVTOKENIZER_CKPT,
        bandwidth_id=0,
        sample_rate=24000,
        max_duration_sec=10.0,
        min_duration_sec=4.0,
        device="cuda",
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.max_seq_len = max_seq_len
        self.sample_rate = sample_rate
        self.bandwidth_id = torch.tensor([bandwidth_id], device=self.device)

        # ---------- load manifest ----------
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        if max_duration_sec is not None:
            filtered = []
            for item in self.data:
                duration = item.get("duration")
                if duration is None:
                    audio_path = item.get("audio_filepath")
                    if not audio_path:
                        continue
                    try:
                        duration = librosa.get_duration(path=audio_path)
                    except Exception:
                        continue
                if duration <= max_duration_sec and duration >= min_duration_sec:
                    filtered.append(item)
            self.data = filtered

        # ---------- load text tokenizer ----------
        self.tokenizer = CharacterTokenizer()
        if os.path.exists(tokenizer_path):
            self.tokenizer.load(tokenizer_path)
        else:
            print("Training CharacterTokenizer...")
            self.tokenizer.train(manifest_path, save_path=tokenizer_path)

        # ---------- load wavtokenizer ----------
        print("Loading WavTokenizer...")
        self.model = WavTokenizer.from_pretrained0802(
            config_path, model_path
        ).to(self.device)
        self.model.eval()

        
    def __len__(self):
        return len(self.data)


    def _load_or_encode(self, audio_path, top_db=30):

        wav_np, sr = librosa.load(audio_path, sr=None, mono=True)
        wav_np, _ = librosa.effects.trim(wav_np, top_db=top_db)
        wav = torch.from_numpy(wav_np).unsqueeze(0)  # (1, T)
        wav = convert_audio(wav, sr, self.sample_rate, 1).to(self.device)

        with torch.no_grad():
            _, codes = self.model.encode_infer(
                wav, bandwidth_id=self.bandwidth_id
            )
            # [n_q, B, T] -> [n_q, T]
            codes = codes.squeeze(1).cpu().numpy()

        return torch.from_numpy(codes).long()

    # --------------------------------------------------
    # __getitem__
    # --------------------------------------------------

    def __getitem__(self, idx):
        item = self.data[idx]
        audio_path = item["audio_filepath"]
        text = item["text"]
    
        # ---------- audio ----------
        codes = self._load_or_encode(audio_path) + self.tokenizer.vocab_size
        audio_seq = codes[0]  # [T]
    
        T = len(audio_seq)
        #split = self._find_energy_split(audio_path, T)

        split = 120 # hardcoded split for 3s @ 40fps, can be replaced by energy-based split like vall-e
    
        audio_in_seq = audio_seq[:split]
        audio_out_seq = audio_seq[split:] 
    
        # ---------- text ----------
        text_ids = self.tokenizer.encode(text)
        text_ids = (
            [self.tokenizer.bos_token_id]
            + text_ids
            + [self.tokenizer.eos_token_id]
        )
        
        text_ids = torch.tensor(text_ids, dtype=torch.long)
        eoa_token = torch.tensor([self.tokenizer.eoa_token_id], dtype=torch.long)

        # ---------- concat ----------
        full_seq = torch.cat(
            [audio_in_seq, audio_out_seq, eoa_token]
        )

        audio_in_len = len(audio_in_seq)
    
        return full_seq, audio_in_len

# --------------------------------------------------
# collate_fn
# --------------------------------------------------
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return torch.empty(0), torch.empty(0), torch.empty(0)

    max_len = max(len(seq) for seq, _ in batch)

    input_ids = torch.full((len(batch), max_len), 0, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -1, dtype=torch.long)

    for i, (seq, audio_in_len) in enumerate(batch):
        L = len(seq)

        # decoder-only shift
        input_ids[i, : L - 1] = seq[:-1]
        targets = seq[1:].clone()

        # mask loss on audio_in
        # targets[k] predicts seq[k+1]
        if audio_in_len > 0:
            targets[: audio_in_len - 1] = -1

        labels[i, : L - 1] = targets

    return input_ids, labels

def pre_training_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return torch.empty(0), torch.empty(0), torch.empty(0)

    max_len = max(len(seq) for seq, _ in batch)

    input_ids = torch.full((len(batch), max_len), 0, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -1, dtype=torch.long)

    for i, (seq, audio_in_len) in enumerate(batch):
        L = len(seq)

        # decoder-only shift
        input_ids[i, : L - 1] = seq[:-1]
        targets = seq[1:].clone()

        # mask loss on audio_in
        # targets[k] predicts seq[k+1]
        #if audio_in_len > 0:
        #    targets[: audio_in_len - 1] = -1

        labels[i, : L - 1] = targets

    return input_ids, labels

# --------------------------------------------------
# dataloader helper
# --------------------------------------------------
def get_dataloader(
    manifest_path,
    batch_size=8,
    max_seq_len=2048,
    shuffle=True,
    num_workers=4,
):
    dataset = SpeechTextSpeechDataset(
        manifest_path=manifest_path,
        max_seq_len=max_seq_len,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
