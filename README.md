<h1 align="center">MELLE Unofficial</h1>

<p align="center">
  <strong>Autoregressive speech synthesis with continuous mel-spectrograms</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2407.08551"><img alt="Paper" src="https://img.shields.io/badge/arXiv-2407.08551-b31b1b.svg"></a>
  <a href="https://github.com/yoospeech/melle_unofficial"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-melle__unofficial-181717?logo=github"></a>
  <a href="https://github.com/gemelo-ai/vocos"><img alt="Vocoder" src="https://img.shields.io/badge/Vocoder-Vocos-6f42c1"></a>
  <img alt="Sample rate" src="https://img.shields.io/badge/Audio-24%20kHz-2ea44f">
  <img alt="Framework" src="https://img.shields.io/badge/PyTorch-BF16-ee4c2c?logo=pytorch&logoColor=white">
</p>

An unofficial PyTorch implementation of
[MELLE](https://arxiv.org/abs/2407.08551). It generates continuous mel frames
autoregressively and uses **Vocos** for both 24 kHz feature extraction and
waveform decoding.

> **Status: Implementation complete** — training, resume, text/prompt
> inference, Vocos decoding, and ASR seed search are supported.

> [!IMPORTANT]
> This is an independent implementation, not the authors' official code or an
> exact reproduction. Its Vocos features differ from the paper's mel setup.

## Implementation choices

Key differences are **Vocos 24 kHz, 100-channel natural-log mel features**, the
pretrained `charactr/vocos-mel-24khz` decoder, a locally trained SentencePiece
character tokenizer, and decoder-only RoPE. Text-only and WAV-prompted
inference are supported; autoregressive decoding currently has no KV cache.

## Overview

```text
Text ──> character tokenizer ──> prefix Transformer decoder ──> continuous mel ──> Vocos ──> 24 kHz audio
                                             │
                                             └── learned mel BOS predicts the first mel frame
```

Text is a bidirectional prefix; mel frames are causal and begin with a learned
`mel_BOS`. Padding and text positions are excluded from acoustic loss.

## Highlights

- Continuous mel autoregression without acoustic quantization
- SentencePiece character tokenizer trained from the manifest
- 12-layer, 1024-dimensional decoder-only Transformer with 16 attention heads
- Variational mel prediction, convolutional post-net, and stop head
- Regression, KL, flux, and stop objectives
- Vocos features/decoder, BF16, DDP, TensorBoard, and checkpoint resume

## Quick start

```bash
git clone --recurse-submodules https://github.com/yoospeech/melle_unofficial.git
cd melle_unofficial

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp manifest.example.json manifest.json
BATCH_SIZE=16 MANIFEST_PATH=manifest.json bash train_melle.sh
```

Edit `manifest.json` with valid audio paths. If Vocos is missing:

```bash
git submodule update --init --recursive
```

## Data

The manifest must be a JSON array containing an audio path and transcript for
each sample:

```json
[
  {
    "audio_filepath": "/absolute/path/to/audio.wav",
    "text": "Nice to meet you.",
    "speaker": "speaker_id",
    "duration": 6.2
  }
]
```

Defaults: 4–10 second clips, 24 kHz resampling, silence trimming at `top_db=40`,
and a deterministic 0.1% validation split. Training uses full-utterance teacher
forcing; prompt conditioning is inference-only. `speaker` is metadata only.

## Acoustic features

The dataset imports `MelSpectrogramFeatures` directly from the included Vocos
submodule.

| Setting | Value |
| --- | ---: |
| Sample rate | 24,000 Hz |
| FFT / window length | 1024 |
| Hop length | 256 |
| Mel channels | 100 |
| Frame rate | 93.75 frames/s |
| Spectrogram power | 1 |
| Padding | Centered |
| Compression | Vocos natural-log `safe_log` |

## Training

Start a single-GPU run:

```bash
BATCH_SIZE=16 \
MANIFEST_PATH=/absolute/path/to/libritts_train_combined_manifest.json \
TOKENIZER_PATH=./melle_character_tokenizer.model \
CUDA_VISIBLE_DEVICES=0 \
bash train_melle.sh
```

The tokenizer is trained automatically if `TOKENIZER_PATH` does not exist.

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `MANIFEST_PATH` | `manifest.json` | Training manifest |
| `TOKENIZER_PATH` | `melle_character_tokenizer.model` | Tokenizer model |
| `BATCH_SIZE` | `16` | Utterances per GPU batch |
| `RESUME_CKPT` | empty | Checkpoint from which to resume |

LR warms to `5e-5` over 32k steps and decays to zero by step 400k. KL weight is
`0` for 10k steps, then `0.1`; flux weight is `0.5`. Losses are normalized by
valid acoustic frames. Outputs are written to:

```text
runs/melle_YYYY_MM_DD_HH_MM_SS/
├── events.out.tfevents...
└── checkpoints/
    └── ckpt.pt
```

Monitor with `tensorboard --logdir runs --bind_all`. Resume with:

```bash
BATCH_SIZE=16 \
MANIFEST_PATH=/absolute/path/to/manifest.json \
TOKENIZER_PATH=./melle_character_tokenizer.model \
RESUME_CKPT=./runs/melle_.../checkpoints/ckpt.pt \
bash train_melle.sh
```

## Pretrained checkpoint

The trained checkpoint and its matching tokenizer are available from
[youspeech/melle-unofficial](https://huggingface.co/youspeech/melle-unofficial):

```bash
hf download youspeech/melle-unofficial \
  ckpt.pt melle_tokenizer.model melle_tokenizer.vocab \
  --local-dir ./checkpoints
```

The downloaded checkpoint also supports `RESUME_CKPT`.

## Inference

### Text only

```bash
bash inference_melle.sh \
  --checkpoint ./checkpoints/ckpt.pt \
  --tokenizer ./checkpoints/melle_tokenizer.model \
  --text "Nice to meet you." \
  --output ./generated.wav
```

### Optional acoustic prompt

`inf_test/` contains the prompt and tokenizer used by `inf_sh.sh`:

```text
inf_test/
├── prompt.wav
├── prompt.txt
├── melle_tokenizer.model
└── melle_tokenizer.vocab
```

`--text` must contain the prompt transcript followed by the continuation:

```bash
CUDA_VISIBLE_DEVICES=0 bash inference_melle.sh \
  --checkpoint ./checkpoints/ckpt.pt \
  --tokenizer ./inf_test/melle_tokenizer.model \
  --prompt-audio ./inf_test/prompt.wav \
  --text "At last the tablecloth was spread there so many genres of movies in there." \
  --output ./inf_test/generated.wav \
  --max-new-seconds 5 \
  --stop-threshold 0.5 \
  --seed 2
```

Useful controls: `--min-new-seconds`, `--max-new-seconds`, `--stop-threshold`,
`--seed`, `--skip-postnet`, and `--prompt-copy-output`.

For optional `faster-whisper` scoring and best-of-N seed search, add:

```bash
  --asr-reference "target text" \
  --asr-backend faster-whisper \
  --asr-model tiny.en \
  --seed 0 \
  --seed-search 5
```

## Repository layout

```text
.
├── melle_{dataset,tokenizer,model,loss}.py
├── train_melle.py / train_melle.sh
├── inference_melle.py / inference_melle.sh
├── inf_test/
├── manifest.example.json
└── vocos/
```

## Current limitations

- No KV cache; long generation is slower.
- Prompting is inference-only; training uses complete utterances.
- Vocos features differ from the paper's 16 kHz, 80-bin log-mel setup.
- The checkpoint is hosted on Hugging Face, not in Git.

## Responsible use

Use only audio and voices for which you have the necessary rights and consent.
Do not use generated speech for impersonation, deception, or misleading
content.

## Acknowledgements

- [MELLE](https://arxiv.org/abs/2407.08551) for the formulation
- [Vocos](https://github.com/gemelo-ai/vocos) for features and decoding
- This implementation referred to [Shy-98/MELLE](https://github.com/Shy-98/MELLE.git).

## Citation

If this repository is useful in your work, please cite the original MELLE
paper:

```bibtex
@article{meng2024melle,
  title   = {Autoregressive Speech Synthesis without Vector Quantization},
  author  = {Meng, Lingwei and Zhou, Long and Liu, Shujie and Chen, Sanyuan and Han, Bing and Hu, Shujie and Liu, Yanqing and Li, Jinyu and Zhao, Sheng and Wu, Xixin and Meng, Helen and Wei, Furu},
  journal = {arXiv preprint arXiv:2407.08551},
  year    = {2024}
}
```
