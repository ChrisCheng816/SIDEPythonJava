# Analysis

当前可执行入口是 `Analysis-SIDE.r`。

默认行为：

- 自动把仓库根目录解析为 `Analysis/..`
- 默认读取 `Results/run-on-test/human-annotated-dataset-with-metrics.csv`
- 自动探测当前仓库中的训练数据和模型目录：
  - `fine-tuning/fine-tuning/train.json`
  - `fine-tuning/fine-tuning/eval.json`
  - `hard-negatives/hard-negatives/`
- 输出写到 `Analysis/output/`

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
