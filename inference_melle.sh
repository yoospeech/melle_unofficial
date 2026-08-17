#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# DGX Spark: prevent Arm CPU thread oversubscription and expose the vendored
# Vocos package without requiring an editable installation.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTHONPATH="${SCRIPT_DIR}/vocos:${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

exec /home/ysw/Documents/anaconda3/envs/vocos_3.13/bin/python \
  "${SCRIPT_DIR}/inference_melle.py" "$@"
