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

An unofficial PyTorch implementation inspired by
[MELLE](https://arxiv.org/abs/2407.08551). The model generates continuous
mel-spectrogram frames autoregressively from text, then converts them to a
24 kHz waveform with [Vocos](https://github.com/gemelo-ai/vocos). Optional
reference-audio prompting is supported during inference.

> [!IMPORTANT]
> This is an independent research implementation, not the authors' official
> code or an exact reproduction. Its 24 kHz Vocos features differ from the
> paper's reported mel configuration. The original negative flux objective is
> also unbounded below; monitor `flux`, `kl`, and total loss during training.

## Overview

```text
Text ──> character tokenizer ──> prefix Transformer ──> continuous mel ──> Vocos ──> 24 kHz audio
                                     │
                                     └── EOS hidden state predicts the first mel frame
```

The training sequence is aligned without a synthetic acoustic `GO` token:

```text
Full sequence    [BOS, text ..., EOS, mel₀, mel₁, ..., melₜ₋₂, melₜ₋₁]
Decoder input    [BOS, text ..., EOS, mel₀, mel₁, ..., melₜ₋₂]
Acoustic queries [                 EOS, mel₀, mel₁, ..., melₜ₋₂]
Mel targets      [                mel₀, mel₁, mel₂, ..., melₜ₋₁]
```

| Sequence region | Attention | Training loss |
| --- | --- | --- |
| Valid text tokens | Fully visible text prefix | Excluded |
| All valid mel frames | Causal over text and earlier mel frames | Included |
| Padding | Blocked as both key and query | Excluded |

## Highlights

- Continuous mel-spectrogram autoregression without acoustic token
  quantization
- SentencePiece character tokenizer trained from the supplied manifest
- 12-layer, 1024-dimensional decoder-only Transformer with 16 attention heads
- Variational latent sampling, convolutional post-net, and learned stop head
- Regression, KL, spectrogram-flux, and stop-prediction objectives
- Direct use of Vocos `MelSpectrogramFeatures` for feature compatibility
- BF16, fused AdamW, gradient clipping, TensorBoard, tqdm, DDP, and resume
  support
- Settings tuned for a single NVIDIA DGX Spark GPU

## Quick start

```bash
git clone --recurse-submodules https://github.com/yoospeech/melle_unofficial.git
cd melle_unofficial

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp manifest.example.json manifest.json
# Edit manifest.json so that each audio_filepath points to a real file.

BATCH_SIZE=16 MANIFEST_PATH=manifest.json bash train_melle.sh
```

If the repository was cloned without submodules:

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

- Samples are filtered to the default duration range of 4–10 seconds.
- Audio is resampled to 24 kHz when necessary.
- Training uses the complete utterance as its mel target; there is no fixed
  three-second acoustic prompt in the training sample.
- After filtering, 0.1% of samples are deterministically reserved for
  validation.
- `speaker` is accepted as manifest metadata but is not currently passed to a
  speaker embedding.

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

The character tokenizer is created automatically from the manifest when the
specified model does not exist. `BATCH_SIZE` is the number of utterances per
GPU step; use 8 if memory is tight, or increase it only after measuring memory
usage.

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `MANIFEST_PATH` | `manifest.json` | Training manifest |
| `TOKENIZER_PATH` | `melle_character_tokenizer.model` | Tokenizer model |
| `BATCH_SIZE` | `16` | Utterances per GPU batch |
| `RESUME_CKPT` | empty | Checkpoint from which to resume |

Default optimization settings include a `5e-5` learning rate, 2,000 warmup
steps, 400,000 maximum steps, and KL annealing to `0.1` over the first 10,000
updates. Runs are written to:

```text
runs/melle_YYYY_MM_DD_HH_MM_SS/
├── events.out.tfevents...
└── checkpoints/
    └── ckpt.pt
```

Monitor a run with TensorBoard:

```bash
tensorboard --logdir runs --bind_all
```

Resume from a checkpoint:

```bash
BATCH_SIZE=16 \
MANIFEST_PATH=/absolute/path/to/manifest.json \
TOKENIZER_PATH=./melle_character_tokenizer.model \
RESUME_CKPT=./runs/melle_.../checkpoints/ckpt.pt \
bash train_melle.sh
```

## Inference

### Text only

Text-only synthesis starts generation from the text `EOS` hidden state, which
matches the first-frame prediction used during training.

```bash
bash inference_melle.sh \
  --checkpoint ./runs/melle_.../checkpoints/ckpt.pt \
  --tokenizer ./melle_character_tokenizer.model \
  --text "Nice to meet you." \
  --output ./generated.wav
```

### Optional acoustic prompt

Pass a reference waveform to condition generation on its leading mel frames.
The text must contain the linguistic context you want the model to condition
on, including the prompt transcript when prompted continuation is intended.

```bash
bash inference_melle.sh \
  --checkpoint ./runs/melle_.../checkpoints/ckpt.pt \
  --tokenizer ./melle_character_tokenizer.model \
  --prompt-audio ./reference.wav \
  --prompt-copy-output ./prompt_copy.wav \
  --text "Prompt transcript. Target sentence." \
  --output ./generated.wav \
  --max-new-seconds 5
```

Useful generation controls include `--min-new-seconds`,
`--max-new-seconds`, `--stop-threshold`, and `--seed`. The default decoder is
the `charactr/vocos-mel-24khz` checkpoint, downloaded on its first use.

## Repository layout

```text
.
├── melle_dataset.py       # Audio loading, Vocos features, collation, masks
├── melle_tokenizer.py     # SentencePiece character tokenizer
├── melle_model.py         # Transformer, latent sampler, post-net, stop head
├── melle_loss.py          # Regression, KL, flux, and stop objectives
├── train_melle.py         # Training, validation, DDP, logs, checkpoints
├── inference_melle.py     # Autoregressive generation and Vocos decoding
├── train_melle.sh         # DGX Spark training launcher
├── inference_melle.sh     # Inference launcher
├── manifest.example.json  # Minimal manifest template
└── vocos/                 # Vocos Git submodule
```

## Current limitations

- Autoregressive inference does not use a KV cache, so long generation is
  computationally expensive.
- Prompted inference is available, but training currently predicts complete
  utterances from text without a separate acoustic-prompt split.
- The Vocos feature scale differs from the paper's reported 16 kHz, 80-bin,
  base-10-log configuration.
- No pretrained MELLE checkpoint is distributed by this repository.

## Responsible use

Use only audio and voices for which you have the necessary rights and consent.
Do not use generated speech for impersonation, deception, or misleading
content.

## Acknowledgements

- [MELLE](https://arxiv.org/abs/2407.08551) for the continuous
  mel-spectrogram language-modeling formulation
- [Vocos](https://github.com/gemelo-ai/vocos) for acoustic feature extraction
  and waveform decoding
Vocos is included as a Git submodule under its own MIT license. Review the
licenses of all dependencies and datasets before redistribution.

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
