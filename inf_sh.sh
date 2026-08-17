#!/bin/bash
set -euo pipefail

# Generate 100 consecutive seeds in one MELLE process, transcribe each
# continuation, and save the candidate with the lowest sCER.
#bash inference_melle.sh \
#  --checkpoint /home/ysw/Documents/eee/melle_unofficial/runs/melle_2026_08_14_22_03_49/checkpoints/ckpt.pt \
#  --tokenizer melle_tokenizer.model \
#  --prompt-audio /home/ysw/Documents/ddd/wavtokenizer_test/8072_284670_000002_000001_0.00s-2.42s.wav \
#  --prompt-copy-output prompt_copy.wav \
#  --text "At last the tablecloth was spread. there are so many genres of movies in there." \
#  --output generated_best.wav \
#  --max-new-seconds 10 \
#  --stop-threshold 0.5 \
#  --seed 101 \

cd /home/ysw/Documents/eee/melle_unofficial

CUDA_VISIBLE_DEVICES=0 ./inference_melle.sh \
  --checkpoint /home/ysw/Documents/eee/melle_unofficial/runs/melle_2026_08_18_06_21_19/checkpoints/ckpt.pt \
  --tokenizer ./melle_tokenizer.model \
  --prompt-audio /home/ysw/Documents/eee/melle_unofficial/8072_284670_000002_000001_0.00s-2.42s.wav \
  --text "At last the tablecloth was spread there so many genres of movies in there." \
  --output ./latest_step_11000.wav \
  --max-new-seconds 5 \
  --stop-threshold 0.5 \
  --seed 10