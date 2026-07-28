# Reproducible Tables

`table2_sidepython.R` produces a Table 2-style SIDE-only comparison. It keeps the
full set of available controls inside each ordered-logit model. The final
publication table keeps every available metric as a row and uses the four human
dimensions as columns. Each cell reports only the odds ratio and p-value. The
Java section contains the legacy and reproduced results; SIDEpython is reported
separately for the no-SIDE and SIDE-filtered conditions.

The checked-in 500-row annotation files provide nine of the ten predictors used in the original Table 2. `ROUGE-4-R` is not present in those files, so the run summary records it as omitted. Supplying a CSV with that column includes it automatically.

Run with the default replay inputs:

```bash
Rscript study-3/scripts/table2_sidepython.R
```

The default inputs and outputs use `2026-04-23-base-hf-side09`. Results are
written to `evaluation/metrics/table2-sidepython/` under that replay root.

Or choose a replay and output directory explicitly:

```bash
Rscript study-3/scripts/table2_sidepython.R \
  --no-side /path/to/no-side_500-human-annotation.csv \
  --with-side /path/to/with-side_500-human-annotation.csv \
  --output-dir /path/to/table2-sidepython
```

`SIDE_score` is normalized to the output label `SIDEpython`. `SIDE_emb` is deliberately not used because it is not the trained Python SIDE score emitted by the project scoring pipeline.
