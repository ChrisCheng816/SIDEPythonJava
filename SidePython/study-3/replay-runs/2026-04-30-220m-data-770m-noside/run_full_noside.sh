#!/usr/bin/env bash
set -euo pipefail

source /home/zcheng06/miniconda3/etc/profile.d/conda.sh
conda activate side-py
cd /scratch/zcheng06/SideJavaPy/SidePython/pipeline

python run_base_replay.py \
  --eval-mode llm \
  --eval-target no-side \
  --skip-data-stages \
  --run-root /scratch/zcheng06/SideJavaPy/SidePython/study-3/replay-runs/2026-04-30-220m-data-770m-noside \
  --model-name-or-path Salesforce/codet5p-770m \
  --tokenizer-name Salesforce/codet5p-770m \
  --cuda-visible-devices 0 \
  --judge-python-bin /home/zcheng06/miniconda3/envs/gptoss-judge/bin/python \
  --judge-cuda-visible-devices 0,1 \
  --train-batch-size 4 \
  --eval-batch-size 4 \
  --gradient-accumulation-steps 4 \
  --skip-train-with-side \
  --skip-infer-with-side
