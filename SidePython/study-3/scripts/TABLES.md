# Reproducible Tables

`table2_sidepython.R` produces the Table 2-style ordered-logit coefficients for the Python extension.
It runs the no-SIDE and SIDE-filtered conditions separately and writes one CSV and one LaTeX table for each available human target: content adequacy, conciseness, and fluency.

The checked-in 500-row annotation files provide nine of the ten predictors used in the original Table 2. `ROUGE-4-R` is not present in those files, so the run summary records it as omitted. Supplying a CSV with that column includes it automatically.

Run with the default replay inputs:

```bash
Rscript study-3/scripts/table2_sidepython.R
```

Or choose a replay and output directory explicitly:

```bash
Rscript study-3/scripts/table2_sidepython.R \
  --no-side /path/to/no-side_500-human-annotation.csv \
  --with-side /path/to/with-side_500-human-annotation.csv \
  --output-dir /path/to/table2-sidepython
```

`SIDE_score` is normalized to the output label `SIDEpython`. `SIDE_emb` is deliberately not used because it is not the trained Python SIDE score emitted by the project scoring pipeline.
