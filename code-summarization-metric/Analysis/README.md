# Analysis

## Table 1 reproduction

`table1.R` is the focused reproduction entry point for the paper's PCA Table 1.
It uses the original `redun(..., r2 = 0.8, nk = 0)` selection followed by an
unscaled PCA, then writes four useful artifacts under `Results/evaluation/table1/`:

- `table1-notebook-reference.csv`: the legacy values recorded in `Analysis-SIDE.ipynb`.
- `table1-reproduced.csv` and `table1-reproduced.tex`: the values from the current input data.
- `table1-notebook-vs-reproduced.csv`: cell-by-cell differences for metrics shared by both tables.
- `table1-reduced-metrics.csv`: the metrics retained by `Hmisc::redun` for the new run.

Run it with the checked-in input or point it at a replacement data file:

```bash
Rscript Analysis/table1.R
Rscript Analysis/table1.R --input /absolute/path/to/human-annotated-dataset-with-metrics.csv
```

当前可执行入口是 `Analysis-SIDE.r`。

默认行为：

- 自动把仓库根目录解析为 `Analysis/..`
- 默认读取 `Results/run-on-test/human-annotated-dataset-with-metrics.csv`
- 自动探测当前仓库中的训练数据和模型目录：
  - `fine-tuning/fine-tuning/train.json`
  - `fine-tuning/fine-tuning/eval.json`
  - `hard-negatives/hard-negatives/`
- 完整分析（含 replication OR 表）写到 `Results/evaluation/analysis/`

运行方式：

```bash
Rscript Analysis/Analysis-SIDE.r
```

也可以显式传入 CSV：

```bash
Rscript Analysis/Analysis-SIDE.r /absolute/path/to/human-annotated-dataset-with-metrics.csv
```

依赖说明：

- `MASS` 是必需依赖，用于 `polr`
- `Hmisc` 和 `xtable` 是可选依赖
- 如果没有 `Hmisc`，脚本会退化为 base R 的相关聚类和简化版冗余筛选，而不是直接失败
- 如果没有 `xtable`，脚本仍会生成 `.csv` 结果，只是不输出 `.tex`

说明：

- `Analysis-SIDE.ipynb` 仍然是历史 notebook；当前工作区优先使用 `Analysis-SIDE.r`
- 脚本会把列名自动规范化到分析所需格式，因此既兼容仓库里的原始 CSV，也兼容 `read.csv` 读入后带点号的旧列名
