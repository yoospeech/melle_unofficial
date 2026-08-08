# MELLE Unofficial

An unofficial PyTorch implementation of MELLE training and inference. A
decoder-only Transformer autoregressively generates continuous mel-spectrogram
frames conditioned on text and an acoustic prompt. Vocos then converts the
generated features into a 24 kHz waveform.

> This repository is an unofficial implementation intended for research and
> experimentation. It is not the authors' official code. Some feature and
> training settings differ from the paper to provide direct Vocos
> compatibility.

## Training structure

```text
Full text ─────────────────┐
                           ├─> Causal Transformer ─> continuous mel ─> Vocos ─> audio
First 3 s acoustic prompt ─┘
```

Each sample is arranged as:

```text
[full text][fixed 3-second prompt mel][continuation mel]
```

| Region | LLM context | Loss |
| --- | --- | --- |
| Real text tokens | Included | Excluded |
| First 3 seconds of prompt mel | Included | Excluded |
| Continuation mel | Included | Included |
| Padding | Excluded | Excluded |

Attention is causal over the complete sequence. Text and prompt features form
the prefix used to generate the continuation. Regression, KL, flux, and stop
losses are applied only to continuation frames.

## Main components

- 4K SentencePiece BPE text tokenizer
- 12-layer decoder-only Transformer
- Model dimension 1024, 16 attention heads, and FFN dimension 4096
- Variational latent sampling module
- Coarse and refined mel regression losses
- KL divergence, spectrogram flux, and stop prediction losses
- Five-layer convolutional post-net
- BF16 and fused AdamW on CUDA
- TensorBoard and tqdm logging
- DDP and checkpoint resume support

## Vocos features

The dataset directly uses `MelSpectrogramFeatures` from
[Vocos](https://github.com/gemelo-ai/vocos).

| Setting | Value |
| --- | ---: |
| Sampling rate | 24,000 Hz |
| FFT/window | 1024 |
| Hop length | 256 |
| Mel channels | 100 |
| Padding | center |
| Magnitude power | 1 |
| Log transform | Vocos `safe_log` |
| Frame rate | 93.75 frames/s |
| Three-second training prompt | 281 frames |

Audio with a different sampling rate is automatically resampled to 24 kHz
with `torchaudio`.

## Installation

```bash
git clone --recurse-submodules https://github.com/yoospeech/melle_unofficial.git
cd melle_unofficial

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If the repository was cloned without submodules, initialize Vocos separately:

```bash
git submodule update --init --recursive
```

## Manifest

The training manifest is a JSON array:

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

- The default duration range is 4–10 seconds.
- Every utterance must be longer than the fixed three-second prompt.
- After filtering, 0.1% of the samples are deterministically reserved for
  validation.
- Manifests and audio files are excluded from Git.

## Training

The following command starts training with a batch size of 16 on a single
NVIDIA DGX Spark GPU:

```bash
BATCH_SIZE=16 \
MANIFEST_PATH=/absolute/path/to/libritts_train_combined_manifest.json \
TOKENIZER_PATH=./melle_tokenizer.model \
CUDA_VISIBLE_DEVICES=0 \
bash train_melle.sh
```

`BATCH_SIZE` is the number of utterances per batch. Reduce it to 8 if memory is
insufficient, or increase it to 32 if enough memory is available.

On the first run, a 4K SentencePiece BPE tokenizer is trained automatically
from the text in the manifest. Runs are stored as:

```text
runs/melle_YYYY_MM_DD_HH_MM_SS/
├── events.out.tfevents...
└── checkpoints/
    └── ckpt.pt
```

Start TensorBoard with:

```bash
tensorboard --logdir runs --bind_all
```

Resume training with:

```bash
BATCH_SIZE=16 \
MANIFEST_PATH=/absolute/path/to/manifest.json \
TOKENIZER_PATH=./melle_tokenizer.model \
RESUME_CKPT=./runs/melle_.../checkpoints/ckpt.pt \
bash train_melle.sh
```

## Inference

Inference uses the complete reference audio file as the acoustic prompt.
Training uses an exact three-second prompt boundary, but an approximately
three-second reference is sufficient for inference.

Pass the full transcript, including both the prompt and continuation content,
through `--text`:

```bash
bash inference_melle.sh \
  --checkpoint ./runs/melle_.../checkpoints/ckpt.pt \
  --tokenizer ./melle_tokenizer.model \
  --prompt-audio ./reference.wav \
  --prompt-copy-output ./prompt_copy.wav \
  --text "Nice to meet you." \
  --output ./generated.wav
```

Generation limits and the stop threshold can also be configured:

```bash
bash inference_melle.sh \
  --checkpoint ./runs/melle_.../checkpoints/ckpt.pt \
  --prompt-audio ./reference.wav \
  --text "Nice to meet you." \
  --output ./generated.wav \
  --min-new-seconds 0.5 \
  --max-new-seconds 5.0 \
  --stop-threshold 0.5 \
  --seed 1337
```

The default vocoder is the official `charactr/vocos-mel-24khz` checkpoint,
which is downloaded automatically during the first inference run. The output
is a 24 kHz WAV containing both the prompt and generated continuation.

## Repository layout

| Path | Purpose |
| --- | --- |
| `melle_dataset.py` | 24 kHz audio processing, Vocos features, and prompt/loss masks |
| `melle_tokenizer.py` | 4K SentencePiece BPE tokenizer |
| `melle_model.py` | Transformer, latent sampler, stop head, and post-net |
| `melle_loss.py` | Regression, KL, flux, and stop losses |
| `train_melle.py` | Training, validation, DDP, logging, and checkpoints |
| `inference_melle.py` | Autoregressive mel generation and Vocos decoding |
| `train_melle.sh` | DGX Spark training environment |
| `inference_melle.sh` | Inference environment |
| `vocos/` | Vocos Git submodule |
| `dataset.py`, `train.py` | Legacy WavTokenizer discrete-token experiments |

## Notes and limitations

- AR inference does not currently use a KV cache, so long generations can be
  slow.
- Vocos uses natural-log 100-bin features. Their scale differs from the
  paper's 16 kHz, 80-bin, base-10-log features.
- The original unbounded negative spectrogram-flux reward is implemented as a
  bounded hinge penalty with a 0.4 frame-level margin, calibrated on the
  Vocos/LibriTTS features used here. KL weight is linearly annealed from 0 to
  0.1 over the first 10K updates.
- Do not use the model to imitate a person's voice without their consent or to
  create misleading content.

## References

- [MELLE: Autoregressive Speech Synthesis without Vector Quantization](https://arxiv.org/abs/2407.08551)
- [Vocos: Closing the Gap Between Time-Domain and Fourier-Based Neural Vocoders](https://github.com/gemelo-ai/vocos)
- [WavTokenizer](https://github.com/jishengpeng/WavTokenizer)
