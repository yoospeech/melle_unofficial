#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# Current MELLE training configuration:
#   16 utterances/GPU, filtered to 4-10 seconds (~6K-15K mel frames/GPU)
#   raw-sum paper losses, peak LR 5e-5, 32K warmup, grad clip 1.0
export CUDA_VISIBLE_DEVICES=0
export MANIFEST_PATH="$(pwd)/libritts_train_combined_manifest.json"
export TOKENIZER_PATH="$(pwd)/melle_tokenizer.model"
export BATCH_SIZE=16

# Keep feature-extraction workers from oversubscribing CPU BLAS threads.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_CACHE_DIR=/tmp/melle_numba_cache

PYTHON=/home/ysw/Documents/anaconda3/envs/vocos_3.13/bin/python

if ! "$PYTHON" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
  echo "CUDA is unavailable in the vocos_3.13 environment; refusing to start CPU training." >&2
  exit 1
fi

exec "$PYTHON" train_melle.py
