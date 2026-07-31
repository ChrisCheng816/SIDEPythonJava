# Reproducible Tables

`table2_sidepython.R` produces the publication Table 2 for the Python study.
It fits one ordered-logit model for each human-rated quality dimension
(content adequacy, conciseness, and fluency). SIDEpython and the other selected
automatic metrics are fitted together, so SIDEpython's odds ratio and p-value
represent its contribution after controlling for the other metrics. It writes
one publication `.tex` file and its matching CSV summary.

The input must be the original human-annotation CSV after every non-SIDE metric
has been recomputed and SIDEpython has been scored. Do not use a CSV where
`codeComment` was replaced by a no-SIDE or SIDE-filtered model prediction,
because its human labels rate a different text.

Run with the default replay inputs:

```bash
Rscript study-3/scripts/table2_sidepython.R
```

The default inputs and outputs use `2026-04-23-base-hf-side09`. Results are
written to `evaluation/metrics/table2-sidepython/` under that replay root.

Or choose a replay and output directory explicitly:

```bash
Rscript study-3/scripts/table2_sidepython.R \
  --input /path/to/human-annotation_with_fresh_side.csv \
  --output-dir /path/to/table2
```

The output has no per-condition or per-dimension `.tex` files.
