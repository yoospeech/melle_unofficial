# VALL-E AR Only

<p align="center">
  <strong>Autoregressive speech modeling with WavTokenizer tokens</strong>
</p>

<p align="center">
  <a href="https://github.com/yoospeech/valle_ar_only"><img src="https://img.shields.io/badge/PyTorch-AR%20Training-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://github.com/jishengpeng/WavTokenizer"><img src="https://img.shields.io/badge/Audio%20Codec-WavTokenizer-4C8BF5" alt="WavTokenizer"></a>
  <a href="https://github.com/yoospeech/valle_ar_only/blob/main/WavTokenizer/LICENSE"><img src="https://img.shields.io/badge/Codec%20License-MIT-green" alt="MIT License"></a>
</p>

This repository contains a compact training pipeline for an autoregressive
speech-token model. Raw audio is converted into discrete 40 token/s codec tokens
with [WavTokenizer](https://github.com/jishengpeng/WavTokenizer), then modeled by
a causal Transformer.

```text
Audio + transcript → WavTokenizer codes → token sequence → AR Transformer
```

> [!IMPORTANT]
> WavTokenizer pretrained weights are required but are not stored in this
> repository. Download the checkpoint before starting training.

## Quick start

```bash
git clone https://github.com/yoospeech/valle_ar_only.git
cd valle_ar_only

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

hf download novateur/WavTokenizer-large-unify-40token \
  wavtokenizer_large_unify_600_24k.ckpt \
  --local-dir WavTokenizer

cp manifest.example.json manifest.json
# Edit manifest.json so that audio_filepath points to your audio files.

MANIFEST_PATH=manifest.json bash train.sh
```

## WavTokenizer setup

A minimal WavTokenizer runtime and its matching 40 token/s configuration are
included under `WavTokenizer/`. Only the pretrained checkpoint must be downloaded
separately from
[Hugging Face](https://huggingface.co/novateur/WavTokenizer-large-unify-40token).
The checkpoint is approximately **1.76 GB**.

```bash
hf download novateur/WavTokenizer-large-unify-40token \
  wavtokenizer_large_unify_600_24k.ckpt \
  --local-dir WavTokenizer
```

If the `hf` command is unavailable:

```bash
pip install -U huggingface-hub
```

Expected layout after download:

```text
WavTokenizer/
├── configs/
│   └── wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml
├── decoder/
├── encoder/
└── wavtokenizer_large_unify_600_24k.ckpt
```

You can verify the downloaded file with:

```bash
sha256sum WavTokenizer/wavtokenizer_large_unify_600_24k.ckpt
```

Expected SHA-256:

```text
72182c1b6bd5ea7f84cf3ec78a0a3244cf42daa660b2e9bce23f5d74064d8205
```

The checkpoint is covered by `.gitignore`; do not commit it to this repository.

## Dataset manifest

Create a JSON array following [manifest.example.json](manifest.example.json):

```json
[
  {
    "audio_filepath": "/absolute/path/to/audio.wav",
    "text": "Example transcript.",
    "speaker": "speaker_id",
    "duration": 6.0
  }
]
```

Each sample must provide `audio_filepath`, `text`, and `duration`. Audio samples
outside the configured duration range are filtered during dataset construction.

## Training

With the checkpoint in its default location:

```bash
MANIFEST_PATH=/path/to/manifest.json bash train.sh
```

With explicit paths or a previous AR checkpoint:

```bash
MANIFEST_PATH=/path/to/manifest.json \
WAVTOKENIZER_CKPT=/path/to/wavtokenizer_large_unify_600_24k.ckpt \
RESUME_CKPT=/path/to/runs/checkpoints/ckpt.pt \
bash train.sh
```

| Environment variable | Default | Description |
| --- | --- | --- |
| `MANIFEST_PATH` | `manifest.json` | Training manifest path |
| `WAVTOKENIZER_CKPT` | `./WavTokenizer/wavtokenizer_large_unify_600_24k.ckpt` | Pretrained codec checkpoint |
| `WAVTOKENIZER_CONFIG` | Included frame40 YAML | Alternate codec configuration |
| `RESUME_CKPT` | Empty | AR checkpoint to resume from |

Training defaults to CUDA. Hyperparameters such as batch size, model dimension,
layer count, learning rate, and duration filters are defined near the top of
[`train.py`](train.py).

## Repository layout

| Path | Purpose |
| --- | --- |
| [`train.py`](train.py) | Training loop, evaluation, logging, and checkpoints |
| [`gpt_decoder.py`](gpt_decoder.py) | Causal Transformer decoder |
| [`dataset.py`](dataset.py) | Audio loading, WavTokenizer encoding, and sequence construction |
| [`tokenizer.py`](tokenizer.py) | Character tokenizer implementation |
| [`tokenizer.json`](tokenizer.json) | Prepared character vocabulary |
| [`WavTokenizer/`](WavTokenizer/) | Minimal codec runtime and configuration |

Generated checkpoints, TensorBoard logs, datasets, model weights, and audio files
are intentionally excluded from Git.

## Experimental MELLE training

The repository also includes a separate continuous mel-spectrogram training
path based on [MELLE (arXiv:2407.08551)](https://arxiv.org/abs/2407.08551).
It keeps the original manifest, tokenizer, DDP, optimizer, logging, and
checkpoint conventions without changing the WavTokenizer training path.

Initialize the vendored Vocos dependency after cloning:

```bash
git submodule update --init --recursive
```

```bash
MANIFEST_PATH=/path/to/manifest.json bash train_melle.sh
```

The MELLE path consists of:

| Path | Purpose |
| --- | --- |
| [`melle_dataset.py`](melle_dataset.py) | Vocos-compatible 24 kHz, 100-bin mel extraction and batching |
| [`melle_model.py`](melle_model.py) | Text/mel pre-nets, causal Transformer, latent sampler, stop head, and post-net |
| [`melle_loss.py`](melle_loss.py) | Regression, KL, spectrogram-flux, and stop losses |
| [`train_melle.py`](train_melle.py) | MELLE training, evaluation, DDP, and checkpoints |

Feature extraction directly reuses the vendored Vocos
`MelSpectrogramFeatures`: 24 kHz audio, a 1024-point transform, 256-sample hop,
100 mel bins, centered padding, magnitude spectrograms, and Vocos `safe_log`.
The MELLE path uses the paper's 4K SentencePiece BPE tokenizer,
1024-wide 12-layer Transformer, 16 attention heads, and a 4096-wide
feed-forward network. The original discrete-token training path retains its
smaller model configuration.

On a single NVIDIA DGX Spark, `train_melle.py` uses BF16, fused AdamW, eight
persistent feature-extraction workers, and unified-memory-aware unpinned
loading. Training uses a fixed batch size of 16 samples and no gradient
accumulation on one Spark. Set `BATCH_SIZE=32` in the environment to change the
batch size. The configuration uses a `5e-5` peak learning rate and 2K-update
warm-up.

The acoustic prompt is fixed to three seconds. This corresponds to 281 Vocos
mel frames at 93.75 frames/s (the original WavTokenizer path uses 120 codec
tokens at 40 tokens/s). Prompt frames remain in the causal LLM context but are
excluded from MELLE's regression, KL, flux, and stop losses.

Training deterministically reserves 0.1% (`VAL_RATIO = 0.001`) of the filtered
manifest as validation data, with at least one validation sample.

After training, run continuation inference with the full transcript and an
approximately three-second reference waveform. Inference uses the complete
reference file as the acoustic prompt; only training fixes the prompt boundary
to exactly three seconds.

```bash
bash inference_melle.sh \
  --checkpoint runs/melle_.../checkpoints/ckpt.pt \
  --tokenizer melle_tokenizer.model \
  --prompt-audio reference.wav \
  --text "Full prompt and continuation transcript." \
  --output generated.wav
```

## Acknowledgements

- Audio tokenization uses
  [WavTokenizer](https://github.com/jishengpeng/WavTokenizer). Its bundled code
  retains the original [MIT license](WavTokenizer/LICENSE).
- [`gpt_decoder.py`](gpt_decoder.py) was developed with reference to Karpathy's
  [`llama2.c/model.py`](https://github.com/karpathy/llama2.c/blob/master/model.py),
  released under the
  [MIT License](https://github.com/karpathy/llama2.c/blob/master/LICENSE).

## Citation

If this repository is useful for your work, please cite WavTokenizer:

```bibtex
@article{ji2024wavtokenizer,
  title={WavTokenizer: An Efficient Acoustic Discrete Codec Tokenizer for Audio Language Modeling},
  author={Ji, Shengpeng and Jiang, Ziyue and Wang, Wen and Chen, Yifu and Fang, Minghui and Zuo, Jialong and Yang, Qian and Cheng, Xize and Wang, Zehan and Li, Ruiqi and others},
  journal={arXiv preprint arXiv:2408.16532},
  year={2024}
}
```
