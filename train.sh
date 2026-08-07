#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:${PWD}/WavTokenizer"
python3 train.py
