# SidePython Pipeline Entrypoints

This folder provides staged one-command entrypoints for the SIDE-py workflow.

## 1) Data preparation

```bash
cd /scratch/zcheng06/SideJavaPy/SidePython
conda run -n side-py python pipeline/run_prepare_codexglue_python.py
conda run -n side-py python pipeline/run_data_prep.py
```

Notes:
- `run_data_prep.py` now supports both `.jsonl` and `.csv` input.
- Default input points to the CodeXGLUE code-to-text Python train split exported from Hugging Face at `study-2/data-files/codexglue-code-to-text/dataset/python/train.jsonl`.
- Data prep keeps a deterministic random 1/2 sample by default (`--sample-ratio 0.5 --seed 42`), then applies the train/validation split.

## 2) Train SIDE-py

```bash
cd /scratch/zcheng06/SideJavaPy/SidePython
conda run -n side-py python pipeline/run_train_sidepy.py \
  --clean-output \
  --cuda-visible-devices 0
```

`--clean-output` removes the existing SIDE-py model directory under `study-2/training-sidep/models/` before training, so the new run replaces the current checkpoint cleanly.

## 3) Score predictions with SIDE-py

```bash
cd /scratch/zcheng06/SidePython/pipeline
python3 run_score_sidepy.py \
  --csv-input /path/to/predictions.csv \
  --checkpoint /path/to/your/sidepy/checkpoint \
  --csv-output /path/to/predictions_with_side.csv \
  --run-text-eval
```

## 4) Compare with other metrics

```bash
cd /scratch/zcheng06/SidePython/pipeline
python3 run_compare_metrics.py \
  --input-csv /path/to/predictions_with_side.csv
```

## Notes

- These entrypoints follow the current repository structure (`study-2` + `study-3`).
- Stage 4 computes BLEU-4, ROUGE-L, METEOR, ChrF, TF-IDF cosine, and SIDE correlation.
- If you need a full "single command all stages" wrapper, add one script that calls these 4 in order.

## Optional one-command wrapper

```bash
cd /scratch/zcheng06/SidePython/pipeline
python3 run_all.py
```

`run_all.py` performs preflight checks (missing Python modules, LFS placeholders) and then runs each stage, skipping blocked stages with explicit reasons.

## Base Replay From Hugging Face

This repo also includes a Base-only replay path that:

- reloads training data from `apcl/funcom-python`
- keeps a raw `no-side` copy
- builds a `with-side` copy filtered with SIDE threshold `0.9`
- downsamples `no-side` with a fixed random seed so each split matches the filtered split size exactly
- reuses the existing trained SIDE-py checkpoint
- writes all outputs to a fresh replay directory under `study-3/replay-runs/`

Traditional-metrics replay. This trains both CodeT5+ 770m models, runs inference, replaces
the 500-row annotation CSV `codeComment` values with model predictions, recomputes the full
Table-2 regression predictor set for each condition, and then computes fresh SIDE with
`codeFunctions/codeComment`. The final condition-specific regression inputs are written as
`evaluation/metrics/<condition>/*_with_fresh_side.csv`; use these files (not the copied
annotation CSVs) when generating Table 2. The run also attaches SIDE scores to the inference
metrics CSVs and writes BLEU-4, ROUGE-L, METEOR, ChrF, TF-IDF, and Spearman-with-SIDE summaries.
The run directory is auto-created as `study-3/replay-runs/YYYY-MM-DD-base-hf-side09`.

```bash
source /home/zcheng06/miniconda3/etc/profile.d/conda.sh
conda activate side-py
cd /scratch/zcheng06/SideJavaPy/SidePython/pipeline
python run_base_replay.py \
  --eval-mode traditional \
  --cuda-visible-devices 0
```

If the filtered `data/hf-with-side-threshold-0_9` and matched
`data/hf-no-side-matched-threshold-0_9` directories already exist under the run root, skip
the data stages and train/evaluate only:

```bash
source /home/zcheng06/miniconda3/etc/profile.d/conda.sh
conda activate side-py
cd /scratch/zcheng06/SideJavaPy/SidePython/pipeline
python run_base_replay.py \
  --eval-mode traditional \
  --skip-data-stages \
  --run-root /scratch/zcheng06/SideJavaPy/SidePython/study-3/replay-runs/YYYY-MM-DD-base-hf-side09 \
  --cuda-visible-devices 0
```

LLM-judge replay. This does the same training and traditional metrics, then runs LLM-as-a-judge
against the fresh-SIDE annotation CSVs.

```bash
source /home/zcheng06/miniconda3/etc/profile.d/conda.sh
conda activate side-py
cd /scratch/zcheng06/SideJavaPy/SidePython/pipeline
python run_base_replay.py \
  --eval-mode llm \
  --cuda-visible-devices 0 \
  --judge-cuda-visible-devices 2,3
```

If traditional replay has already finished and you only want to add LLM-as-a-judge later, run:

```bash
source /home/zcheng06/miniconda3/etc/profile.d/conda.sh
conda activate side-py
cd /scratch/zcheng06/SideJavaPy/SidePython/pipeline
python run_base_replay.py \
  --eval-mode llm-only \
  --run-root /scratch/zcheng06/SideJavaPy/SidePython/study-3/replay-runs/YYYY-MM-DD-base-hf-side09 \
  --judge-cuda-visible-devices 2,3
```

Useful stage-by-stage commands:

```bash
python run_base_data_prep_hf.py
python run_base_side_filter.py
python run_train_base_replay.py --train-file ... --valid-file ... --test-file ... --base-output-dir ...
python run_infer_base_replay.py --input-file ... --model-path ... --output-dir ...
python run_replace_codecomment_from_inference.py --annotation-csv ... --inference-csv ...
```
