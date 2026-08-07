#!/usr/bin/env bash
set -euo pipefail

# Eight DataLoader processes handle feature extraction. Keep their underlying
# BLAS/OpenMP libraries from oversubscribing the DGX Spark's 20 Arm CPU cores.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

python3 train_melle.py
