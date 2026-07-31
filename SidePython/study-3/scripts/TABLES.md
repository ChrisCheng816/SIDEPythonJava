# Reproducible Tables

`table2_sidepython.R` produces the publication Table 2 for SIDEpython only.
It fits one ordered-logit model for each human-rated quality dimension
(content adequacy, conciseness, and fluency). Each cell reports SIDEpython's
odds ratio and p-value. It writes one publication `.tex` file and its matching
CSV summary.

The input must be the original human-annotation CSV after SIDEpython has been
scored. Do not use a CSV where `codeComment` was replaced by a no-SIDE or
SIDE-filtered model prediction, because its human labels rate a different text.

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
