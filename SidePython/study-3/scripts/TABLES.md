# Reproducible Tables

`table2_sidepython.R` produces a Table 2-style SIDE-only comparison. It keeps the
full set of available controls inside each ordered-logit model. The final
publication table keeps every available metric as a row and uses the four human
dimensions as columns. Each cell reports only the odds ratio and p-value. The
Java section contains the legacy and reproduced results; SIDEpython is reported
separately for the no-SIDE and SIDE-filtered conditions.

The table must use the condition-specific `*_with_fresh_side.csv` files produced by `run_base_replay.py`, not the copied 500-row annotation CSVs. The replay first replaces `codeComment`, recomputes every non-SIDE predictor, and then adds fresh SIDE scores. This prevents the two conditions from silently sharing source-annotation metrics.

Run with the default replay inputs:

```bash
Rscript study-3/scripts/table2_sidepython.R
```

The default inputs and outputs use `2026-04-23-base-hf-side09`. Results are
written to `evaluation/metrics/table2-sidepython/` under that replay root.

Or choose a replay and output directory explicitly:

```bash
Rscript study-3/scripts/table2_sidepython.R \
  --no-side /path/to/no-side_500-human-annotation_with_fresh_side.csv \
  --with-side /path/to/with-side_500-human-annotation_with_fresh_side.csv \
  --output-dir /path/to/table2-sidepython
```

`SIDE_score` is normalized to the output label `SIDEpython`. `SIDE_emb` is deliberately not used because it is not the trained Python SIDE score emitted by the project scoring pipeline.
