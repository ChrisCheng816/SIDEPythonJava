# Python SIDE Tables

`table2_sidepython.R` writes both Python extension tables from the same
human-annotated CSV containing a fresh `SIDE_score`.

- **Table 2** is a multivariable ordered-logit analysis of Content Adequacy,
  Conciseness, and Fluency. Its nine predictors are min-max scaled to 0--5,
  matching the Java SIDE Table 2 procedure. `ROUGE-4-R` is excluded from all
  three models. The table reports unadjusted Wald p-values.
- **Table 3** is unscaled PCA on the same nine predictors, matching Java
  Table 1's PCA step.

```bash
Rscript study-3/scripts/table2_sidepython.R \
  --input /path/to/human-annotation_with_fresh_side.csv \
  --output-dir /path/to/table-output
```

The outputs are `table2-sidepython-regression.tex` and
`table3-sidepython-pca.tex`; both contain only a `tabular` environment.
