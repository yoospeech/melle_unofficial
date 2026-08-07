# MELLE Unofficial

PyTorch로 구현한 비공식 MELLE 학습 및 추론 코드입니다. Decoder-only
Transformer가 text와 acoustic prompt를 조건으로 연속적인 mel-spectrogram
frame을 autoregressive하게 생성하고, 생성된 feature는 Vocos를 통해 24 kHz
waveform으로 복원합니다.

> 이 repository는 연구 및 실험 목적의 비공식 구현입니다. 원 논문의 공식
> 코드가 아니며, Vocos 호환성을 위해 일부 feature와 학습 설정이 논문과
> 다릅니다.

## 학습 구조

```text
전체 text ───────────────┐
                         ├─> Causal Transformer ─> continuous mel ─> Vocos ─> audio
앞 3초 acoustic prompt ──┘
```

각 sample은 다음 순서로 학습됩니다.

```text
[전체 text][고정 3초 prompt mel][continuation mel]
```

| 구간 | LLM context | Loss |
| --- | --- | --- |
| 실제 text token | 포함 | 제외 |
| 앞 3초 prompt mel | 포함 | 제외 |
| continuation mel | 포함 | 포함 |
| padding | 제외 | 제외 |

전체 attention은 causal입니다. Text와 prompt는 continuation을 생성하기 위한
prefix로 사용되며, regression, KL, flux 및 stop loss는 continuation 구간에만
적용됩니다.

## 주요 구성

- 4K SentencePiece BPE text tokenizer
- 12-layer decoder-only Transformer
- Model dimension 1024, 16 attention heads, FFN dimension 4096
- Variational latent sampling module
- Coarse/final mel regression loss
- KL divergence, spectrogram flux 및 stop prediction loss
- 5-layer convolutional post-net
- BF16 및 fused AdamW
- TensorBoard와 tqdm logging
- DDP 및 checkpoint resume 지원

## Vocos feature

Dataset은 [Vocos](https://github.com/gemelo-ai/vocos)의
`MelSpectrogramFeatures`를 직접 사용합니다.

| 설정 | 값 |
| --- | ---: |
| Sampling rate | 24,000 Hz |
| FFT/window | 1024 |
| Hop length | 256 |
| Mel channels | 100 |
| Padding | center |
| Magnitude power | 1 |
| Log transform | Vocos `safe_log` |
| Frame rate | 93.75 frames/s |
| 3초 학습 prompt | 281 frames |

입력 음성이 24 kHz가 아니면 `torchaudio`를 통해 자동 resampling됩니다.

## 설치

```bash
git clone --recurse-submodules https://github.com/yoospeech/melle_unofficial.git
cd melle_unofficial

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

이미 clone했다면 Vocos submodule을 별도로 초기화합니다.

```bash
git submodule update --init --recursive
```

## Manifest

학습 manifest는 JSON array 형식입니다.

```json
[
  {
    "audio_filepath": "/absolute/path/to/audio.wav",
    "text": "만나서 반갑습니다.",
    "speaker": "speaker_id",
    "duration": 6.2
  }
]
```

- 기본 duration 범위는 4–10초입니다.
- 음성은 고정 3초 prompt보다 길어야 합니다.
- filtering 이후 0.1%를 validation으로 deterministic하게 분리합니다.
- Manifest와 audio 파일은 Git에 포함되지 않습니다.

## 학습

DGX Spark 단일 GPU에서 batch size 16으로 시작하는 예시입니다.

```bash
BATCH_SIZE=16 \
MANIFEST_PATH=/absolute/path/to/libritts_train_combined_manifest.json \
TOKENIZER_PATH=./melle_tokenizer.model \
CUDA_VISIBLE_DEVICES=0 \
bash train_melle.sh
```

`BATCH_SIZE`는 sample 개수입니다. 메모리가 부족하면 8로 낮추고, 여유가
있으면 32로 높일 수 있습니다.

첫 실행에서는 manifest의 text로 4K SentencePiece BPE tokenizer를 자동
생성합니다. Run은 다음 경로에 저장됩니다.

```text
runs/melle_YYYY_MM_DD_HH_MM_SS/
├── events.out.tfevents...
└── checkpoints/
    └── ckpt.pt
```

TensorBoard:

```bash
tensorboard --logdir runs --bind_all
```

학습 재개:

```bash
BATCH_SIZE=16 \
MANIFEST_PATH=/absolute/path/to/manifest.json \
TOKENIZER_PATH=./melle_tokenizer.model \
RESUME_CKPT=./runs/melle_.../checkpoints/ckpt.pt \
bash train_melle.sh
```

## 추론

추론에서는 reference audio 파일 전체를 acoustic prompt로 사용합니다. 학습은
정확히 3초 prompt로 수행하지만, 추론 prompt는 대략 3초면 충분합니다.

`--text`에는 prompt와 생성 구간을 포함한 전체 transcript를 전달합니다.

```bash
bash inference_melle.sh \
  --checkpoint ./runs/melle_.../checkpoints/ckpt.pt \
  --tokenizer ./melle_tokenizer.model \
  --prompt-audio ./reference.wav \
  --text "만나서 반갑습니다." \
  --output ./generated.wav
```

생성 길이와 stop threshold를 지정할 수도 있습니다.

```bash
bash inference_melle.sh \
  --checkpoint ./runs/melle_.../checkpoints/ckpt.pt \
  --prompt-audio ./reference.wav \
  --text "만나서 반갑습니다." \
  --output ./generated.wav \
  --min-new-seconds 0.5 \
  --max-new-seconds 5.0 \
  --stop-threshold 0.5 \
  --seed 1337
```

기본 vocoder는 공식 `charactr/vocos-mel-24khz` checkpoint이며 최초 추론 시
자동 다운로드됩니다. 출력은 prompt와 생성 continuation을 포함한 24 kHz
WAV 파일입니다.

## 파일 구성

| 경로 | 역할 |
| --- | --- |
| `melle_dataset.py` | 24 kHz audio 처리, Vocos feature 추출, prompt/loss mask |
| `melle_tokenizer.py` | 4K SentencePiece BPE tokenizer |
| `melle_model.py` | Transformer, latent sampler, stop head 및 post-net |
| `melle_loss.py` | Regression, KL, flux 및 stop loss |
| `train_melle.py` | 학습, validation, DDP, logging 및 checkpoint |
| `inference_melle.py` | Autoregressive mel 생성 및 Vocos decoding |
| `train_melle.sh` | DGX Spark 학습 실행 환경 |
| `inference_melle.sh` | 추론 실행 환경 |
| `vocos/` | Vocos Git submodule |
| `dataset.py`, `train.py` | 기존 WavTokenizer discrete-token 실험 코드 |

## 주의 사항

- 현재 AR inference는 KV cache를 사용하지 않으므로 긴 생성은 느릴 수 있습니다.
- Vocos의 natural-log 100-bin feature를 사용하므로 원 논문의 16 kHz,
  80-bin log10 feature와 loss scale이 동일하지 않습니다.
- Spectrogram flux는 음수 reward 형태이므로 regression과 KL 추세를 함께
  모니터링해야 합니다.
- 승인받지 않은 사람의 음성을 모방하거나 오인 가능성이 있는 콘텐츠를
  생성하는 용도로 사용하지 마십시오.

## References

- [MELLE: Autoregressive Speech Synthesis without Vector Quantization](https://arxiv.org/abs/2407.08551)
- [Vocos: Closing the Gap Between Time-Domain and Fourier-Based Neural Vocoders](https://github.com/gemelo-ai/vocos)
- [WavTokenizer](https://github.com/jishengpeng/WavTokenizer)
