get_script_dir <- function() {
  cmd_args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", cmd_args, value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg[1]))))
  }
  normalizePath(getwd())
}

find_first_existing <- function(paths) {
  normalized <- Filter(
    f = function(path) !is.na(path) && nzchar(path),
    x = paths
  )
  for (path in normalized) {
    if (file.exists(path)) {
      return(normalizePath(path, mustWork = TRUE))
    }
  }
  NULL
}

normalize_columns <- function(dataframe) {
  rename_map <- c(
    "Overall.DA.Score" = "Overall DA Score",
    "Content.Adequacy" = "Content Adequacy",
    "Jaccard" = "Jaccard",
    "jaccard" = "Jaccard",
    "BLEU-A" = "BLEU-A",
    "bleu-A" = "BLEU-A",
    "bleu.A" = "BLEU-A",
    "BLEU-1" = "BLEU-1",
    "bleu-1" = "BLEU-1",
    "bleu.1" = "BLEU-1",
    "BLEU-2" = "BLEU-2",
    "bleu-2" = "BLEU-2",
    "bleu.2" = "BLEU-2",
    "BLEU-3" = "BLEU-3",
    "bleu-3" = "BLEU-3",
    "bleu.3" = "BLEU-3",
    "BLEU-4" = "BLEU-4",
    "bleu-4" = "BLEU-4",
    "bleu.4" = "BLEU-4",
    "TF_IDF_CS" = "TF_IDF_CS",
    "tfidf_cosine" = "TF_IDF_CS",
    "USE_CS" = "USE_CS",
    "USE_cosine_similarity" = "USE_CS",
    "BERTScore-P" = "BERTScore-P",
    "bert-score-precision" = "BERTScore-P",
    "bert.score.precision" = "BERTScore-P",
    "BERTScore-R" = "BERTScore-R",
    "bert-score-recall" = "BERTScore-R",
    "bert.score.recall" = "BERTScore-R",
    "BERTScore-F1" = "BERTScore-F1",
    "bert-score-f1" = "BERTScore-F1",
    "bert.score.f1" = "BERTScore-F1",
    "SentenceBERT_CS" = "SentenceBERT_CS",
    "sentence_bert_cosine_similarity" = "SentenceBERT_CS",
    "InferSent_CS" = "InferSent_CS",
    "infersent_cosine_similarity" = "InferSent_CS",
    "METEOR" = "METEOR",
    "meteor" = "METEOR",
    "ROUGE-L-F1" = "ROUGE-L-F1",
    "rouge-l-f1" = "ROUGE-L-F1",
    "rouge.l.f1" = "ROUGE-L-F1",
    "ROUGE-1-F1" = "ROUGE-1-F1",
    "rouge-1-f1" = "ROUGE-1-F1",
    "rouge.1.f1" = "ROUGE-1-F1",
    "ROUGE-1-P" = "ROUGE-1-P",
    "rouge-1-p" = "ROUGE-1-P",
    "rouge.1.p" = "ROUGE-1-P",
    "ROUGE-1-R" = "ROUGE-1-R",
    "rouge-1-r" = "ROUGE-1-R",
    "rouge.1.r" = "ROUGE-1-R",
    "ROUGE-2-F1" = "ROUGE-2-F1",
    "rouge-2-f1" = "ROUGE-2-F1",
    "rouge.2.f1" = "ROUGE-2-F1",
    "ROUGE-2-P" = "ROUGE-2-P",
    "rouge-2-p" = "ROUGE-2-P",
    "rouge.2.p" = "ROUGE-2-P",
    "ROUGE-2-R" = "ROUGE-2-R",
    "rouge-2-r" = "ROUGE-2-R",
    "rouge.2.r" = "ROUGE-2-R",
    "ROUGE-3-F1" = "ROUGE-3-F1",
    "rouge-3-f1" = "ROUGE-3-F1",
    "rouge.3.f1" = "ROUGE-3-F1",
    "ROUGE-3-P" = "ROUGE-3-P",
    "rouge-3-p" = "ROUGE-3-P",
    "rouge.3.p" = "ROUGE-3-P",
    "ROUGE-3-R" = "ROUGE-3-R",
    "rouge-3-r" = "ROUGE-3-R",
    "rouge.3.r" = "ROUGE-3-R",
    "ROUGE-4-F1" = "ROUGE-4-F1",
    "rouge-4-f1" = "ROUGE-4-F1",
    "rouge.4.f1" = "ROUGE-4-F1",
    "ROUGE-4-P" = "ROUGE-4-P",
    "rouge-4-p" = "ROUGE-4-P",
    "rouge.4.p" = "ROUGE-4-P",
    "ROUGE-4-R" = "ROUGE-4-R",
    "rouge-4-r" = "ROUGE-4-R",
    "rouge.4.r" = "ROUGE-4-R",
    "ROUGE-L-P" = "ROUGE-L-P",
    "rouge-l-p" = "ROUGE-L-P",
    "rouge.l.p" = "ROUGE-L-P",
    "ROUGE-L-R" = "ROUGE-L-R",
    "rouge-l-r" = "ROUGE-L-R",
    "rouge.l.r" = "ROUGE-L-R",
    "ROUGE-W-F1" = "ROUGE-W-F1",
    "rouge-w-f1" = "ROUGE-W-F1",
    "rouge.w.f1" = "ROUGE-W-F1",
    "ROUGE-W-P" = "ROUGE-W-P",
    "rouge-w-p" = "ROUGE-W-P",
    "rouge.w.p" = "ROUGE-W-P",
    "ROUGE-W-R" = "ROUGE-W-R",
    "rouge-w-r" = "ROUGE-W-R",
    "rouge.w.r" = "ROUGE-W-R",
    "TF_IDF_ED" = "TF_IDF_ED",
    "tfidf_euclidean" = "TF_IDF_ED",
    "chrF" = "chrF",
    "chrf_score" = "chrF",
    "USE_ED" = "USE_ED",
    "USE_euclidean_distance" = "USE_ED",
    "SentenceBERT_ED" = "SentenceBERT_ED",
    "sentence_bert_euclidean_distance" = "SentenceBERT_ED",
    "InferSent_ED" = "InferSent_ED",
    "infersent_euclidean_distance" = "InferSent_ED",
    "CodeT5-plus_CS" = "CodeT5-plus_CS",
    "CodeT5-plus-cosine-similarity" = "CodeT5-plus_CS",
    "CodeT5.plus.cosine.similarity" = "CodeT5-plus_CS",
    "SIDE_HARD" = "SIDE",
    "SIDE" = "SIDE",
    "SIDE_TRIVIAL" = "SIDE Trivial",
    "SIDE Trivial" = "SIDE Trivial"
  )

  current_names <- colnames(dataframe)
  current_names[current_names == ""] <- "X"
  for (old_name in names(rename_map)) {
    current_names[current_names == old_name] <- rename_map[[old_name]]
  }
  colnames(dataframe) <- current_names
  dataframe
}

report_package <- function(package_name, required = FALSE) {
  available <- requireNamespace(package_name, quietly = TRUE)
  state <- if (available) "available" else if (required) "missing (required)" else "missing (optional)"
  message(sprintf("[check] package %-8s : %s", package_name, state))
  if (required && !available) {
    stop(
      sprintf("Missing required R package '%s'.", package_name),
      call. = FALSE
    )
  }
  available
}

resolve_assets <- function(repo_root) {
  model_root <- find_first_existing(c(
    file.path(repo_root, "hard-negatives", "hard-negatives"),
    file.path(repo_root, "models", "triplet-loss", "hard_negatives")
  ))

  train_json <- find_first_existing(c(
    file.path(repo_root, "fine-tuning", "fine-tuning", "train.json"),
    file.path(repo_root, "dataset", "train.json")
  ))

  eval_json <- find_first_existing(c(
    file.path(repo_root, "fine-tuning", "fine-tuning", "eval.json"),
    file.path(repo_root, "dataset", "eval.json")
  ))

  list(
    model_root = model_root,
    train_json = train_json,
    eval_json = eval_json
  )
}

resolve_input_csv <- function(repo_root, cli_input) {
  find_first_existing(c(
    cli_input,
    file.path(repo_root, "Results", "run-on-test", "human-annotated-dataset-with-metrics.csv"),
    file.path(repo_root, "human-annotated-dataset-with-metrics.csv"),
    file.path(repo_root, "Results", "run-on-test", "human-annotated-dataset-all-metrics.csv")
  ))
}

validate_columns <- function(dataframe, required_columns) {
  missing_columns <- setdiff(required_columns, colnames(dataframe))
  if (length(missing_columns) > 0) {
    stop(
      sprintf(
        "Input CSV is missing required columns: %s",
        paste(missing_columns, collapse = ", ")
      ),
      call. = FALSE
    )
  }
}

scale_to_range <- function(values, upper_bound) {
  min_value <- min(values, na.rm = TRUE)
  max_value <- max(values, na.rm = TRUE)
  if (isTRUE(all.equal(min_value, max_value))) {
    return(rep(0, length(values)))
  }
  (values - min_value) * upper_bound / (max_value - min_value)
}

save_correlation_csv <- function(matrix_data, output_path) {
  utils::write.csv(
    data.frame(metric = rownames(matrix_data), matrix_data, check.names = FALSE),
    output_path,
    row.names = FALSE
  )
}

save_text_output <- function(lines, output_path) {
  writeLines(lines, con = output_path)
}

save_cluster_plot <- function(correlation_matrix, output_path, use_hmisc, metric_frame) {
  grDevices::pdf(output_path, width = 12, height = 8)
  on.exit(grDevices::dev.off(), add = TRUE)

  if (use_hmisc) {
    cluster_object <- Hmisc::varclus(
      as.matrix(metric_frame),
      similarity = "spearman",
      type = "data.matrix"
    )
    plot(cluster_object, main = "Metric Clusters (Hmisc::varclus)")
    return(cluster_object)
  }

  clustering <- stats::hclust(
    stats::as.dist(1 - abs(correlation_matrix)),
    method = "average"
  )
  plot(
    clustering,
    main = "Metric Clusters (fallback: average linkage on |Spearman|)",
    xlab = "",
    sub = ""
  )
  clustering
}

select_reduced_metrics <- function(metric_frame, eval_metrics, use_hmisc) {
  formula_text <- paste(sprintf("`%s`", eval_metrics), collapse = " + ")
  formula_object <- stats::as.formula(paste("~", formula_text))

  if (use_hmisc) {
    redundancy <- Hmisc::redun(
      formula_object,
      data = metric_frame,
      r2 = 0.8,
      nk = 0
    )
    reduced_metrics <- redundancy$In
    reduced_metrics <- reduced_metrics[!is.na(reduced_metrics)]
    return(unname(reduced_metrics))
  }

  correlation_matrix <- abs(stats::cor(
    metric_frame[, eval_metrics, drop = FALSE],
    method = "spearman",
    use = "pairwise.complete.obs"
  ))
  cutoff <- sqrt(0.8)
  reduced_metrics <- character()

  for (metric in eval_metrics) {
    if (length(reduced_metrics) == 0) {
      reduced_metrics <- c(reduced_metrics, metric)
      next
    }

    if (all(correlation_matrix[metric, reduced_metrics] < cutoff, na.rm = TRUE)) {
      reduced_metrics <- c(reduced_metrics, metric)
    }
  }

  reduced_metrics
}

fit_polr_table <- function(dataframe, dependent_metric, predictors, scale_upper_bound) {
  model_frame <- dataframe
  for (metric in predictors) {
    model_frame[[metric]] <- scale_to_range(model_frame[[metric]], scale_upper_bound)
  }

  model_frame[[dependent_metric]] <- ordered(round(model_frame[[dependent_metric]]))
  formula_text <- paste(sprintf("`%s`", predictors), collapse = " + ")
  formula_object <- stats::as.formula(
    sprintf("`%s` ~ %s", dependent_metric, formula_text)
  )

  model <- MASS::polr(formula_object, data = model_frame, Hess = TRUE)
  coefficients <- coef(summary(model))
  predictor_rows <- seq_len(length(predictors))
  coefficients <- coefficients[predictor_rows, , drop = FALSE]

  p_values <- 2 * stats::pnorm(abs(coefficients[, "t value"]), lower.tail = FALSE)

  data.frame(
    metric = rownames(coefficients),
    OR = exp(coefficients[, "Value"]),
    Value = coefficients[, "Value"],
    `Std. Error` = coefficients[, "Std. Error"],
    `t value` = coefficients[, "t value"],
    `p value` = round(p_values, 3),
    check.names = FALSE,
    row.names = NULL
  )
}

write_table_outputs <- function(table_data, csv_path, tex_path = NULL) {
  utils::write.csv(table_data, csv_path, row.names = FALSE)
  if (!is.null(tex_path) && requireNamespace("xtable", quietly = TRUE)) {
    print(
      xtable::xtable(table_data),
      file = tex_path,
      include.rownames = FALSE
    )
  }
}

args <- commandArgs(trailingOnly = TRUE)
script_dir <- get_script_dir()
repo_root <- normalizePath(file.path(script_dir, ".."), mustWork = TRUE)
output_dir <- file.path(repo_root, "Results", "evaluation", "analysis")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

invisible(report_package("MASS", required = TRUE))
has_hmisc <- report_package("Hmisc", required = FALSE)
has_xtable <- report_package("xtable", required = FALSE)

assets <- resolve_assets(repo_root)
input_csv <- resolve_input_csv(repo_root, if (length(args) >= 1) args[1] else NA_character_)

if (is.null(input_csv)) {
  stop(
    paste(
      "Could not find an input CSV.",
      "Expected one of:",
      file.path(repo_root, "Results", "run-on-test", "human-annotated-dataset-with-metrics.csv"),
      file.path(repo_root, "human-annotated-dataset-with-metrics.csv")
    ),
    call. = FALSE
  )
}

message(sprintf("[check] repo root          : %s", repo_root))
message(sprintf("[check] input csv          : %s", input_csv))
message(sprintf("[check] train dataset      : %s", ifelse(is.null(assets$train_json), "not found", assets$train_json)))
message(sprintf("[check] eval dataset       : %s", ifelse(is.null(assets$eval_json), "not found", assets$eval_json)))
message(sprintf("[check] model root         : %s", ifelse(is.null(assets$model_root), "not found", assets$model_root)))
message(sprintf("[check] output dir         : %s", output_dir))

t <- utils::read.csv(input_csv, check.names = FALSE)
t <- normalize_columns(t)

user_metrics <- c("Overall DA Score", "Content Adequacy", "Conciseness", "Fluency")
eval_metrics <- c(
  "Jaccard", "BLEU-A", "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "TF_IDF_CS",
  "USE_CS", "BERTScore-P", "BERTScore-R", "BERTScore-F1", "SentenceBERT_CS",
  "InferSent_CS", "METEOR", "ROUGE-L-F1", "c_coeff", "ROUGE-1-F1", "ROUGE-1-P",
  "ROUGE-1-R", "ROUGE-2-F1", "ROUGE-2-P", "ROUGE-2-R", "ROUGE-3-F1", "ROUGE-3-P",
  "ROUGE-3-R", "ROUGE-4-F1", "ROUGE-4-P", "ROUGE-4-R", "ROUGE-L-P", "ROUGE-L-R",
  "ROUGE-W-F1", "ROUGE-W-P", "ROUGE-W-R", "TF_IDF_ED", "chrF", "USE_ED",
  "SentenceBERT_ED", "InferSent_ED", "CodeT5-plus_CS", "SIDE"
)

validate_columns(t, c(user_metrics, eval_metrics, "mid"))

tsum <- subset(t, mid != 0)
if (nrow(tsum) == 0) {
  stop("No rows remain after filtering out records with mid == 0.", call. = FALSE)
}

user_metric_frame <- tsum[, user_metrics, drop = FALSE]
eval_metric_frame <- tsum[, eval_metrics, drop = FALSE]

user_metric_correlations <- stats::cor(
  user_metric_frame,
  method = "spearman",
  use = "pairwise.complete.obs"
)
eval_metric_correlations <- stats::cor(
  eval_metric_frame,
  method = "spearman",
  use = "pairwise.complete.obs"
)

invisible(save_cluster_plot(
  correlation_matrix = eval_metric_correlations,
  output_path = file.path(output_dir, "metric-clusters.pdf"),
  use_hmisc = has_hmisc,
  metric_frame = eval_metric_frame
))

reduced_metrics <- select_reduced_metrics(tsum, eval_metrics, has_hmisc)
if (length(reduced_metrics) == 0) {
  stop("Could not derive a non-empty reduced metric set.", call. = FALSE)
}

pca_result <- stats::prcomp(tsum[, reduced_metrics, drop = FALSE], scale. = FALSE)
overall_score <- (tsum[["Content Adequacy"]] + tsum[["Fluency"]] + tsum[["Conciseness"]]) / 3
overall_correlation <- stats::cor.test(
  overall_score,
  tsum[["Overall DA Score"]],
  method = "spearman"
)

regression_targets <- list(
  list(name = "Overall DA Score", upper = 100),
  list(name = "Content Adequacy", upper = 5),
  list(name = "Conciseness", upper = 5),
  list(name = "Fluency", upper = 5)
)

regression_tables <- list()
for (target in regression_targets) {
  regression_tables[[target$name]] <- fit_polr_table(
    dataframe = tsum,
    dependent_metric = target$name,
    predictors = reduced_metrics,
    scale_upper_bound = target$upper
  )
}

overall_frame <- tsum
overall_frame[["Overall"]] <- overall_score
regression_tables[["Overall"]] <- fit_polr_table(
  dataframe = overall_frame,
  dependent_metric = "Overall",
  predictors = reduced_metrics,
  scale_upper_bound = 5
)

save_correlation_csv(
  user_metric_correlations,
  file.path(output_dir, "user-metric-spearman.csv")
)
save_correlation_csv(
  eval_metric_correlations,
  file.path(output_dir, "eval-metric-spearman.csv")
)
utils::write.csv(
  data.frame(metric = reduced_metrics, check.names = FALSE),
  file.path(output_dir, "reduced-metrics.csv"),
  row.names = FALSE
)
utils::write.csv(
  data.frame(metric = rownames(pca_result$rotation), pca_result$rotation, check.names = FALSE),
  file.path(output_dir, "pca-rotation.csv"),
  row.names = FALSE
)
utils::write.csv(
  data.frame(component = rownames(summary(pca_result)$importance), summary(pca_result)$importance, check.names = FALSE),
  file.path(output_dir, "pca-importance.csv"),
  row.names = FALSE
)

for (target_name in names(regression_tables)) {
  base_name <- gsub("[^A-Za-z0-9]+", "-", tolower(target_name))
  write_table_outputs(
    regression_tables[[target_name]],
    csv_path = file.path(output_dir, sprintf("%s-polr.csv", base_name)),
    tex_path = if (has_xtable) file.path(output_dir, sprintf("%s-polr.tex", base_name)) else NULL
  )
}

summary_lines <- capture.output({
  cat("Repository root:", repo_root, "\n")
  cat("Input CSV:", input_csv, "\n")
  cat("Train JSON:", ifelse(is.null(assets$train_json), "not found", assets$train_json), "\n")
  cat("Eval JSON:", ifelse(is.null(assets$eval_json), "not found", assets$eval_json), "\n")
  cat("Model root:", ifelse(is.null(assets$model_root), "not found", assets$model_root), "\n")
  cat("Hmisc available:", has_hmisc, "\n")
  cat("xtable available:", has_xtable, "\n")
  cat("Rows before filtering:", nrow(t), "\n")
  cat("Rows after filtering (mid != 0):", nrow(tsum), "\n")
  cat("Reduced metrics:", paste(reduced_metrics, collapse = ", "), "\n\n")
  cat("Summary of input dataset:\n")
  print(summary(t))
  cat("\nSummary after filtering mid != 0:\n")
  print(summary(tsum))
  cat("\nOverall vs Overall DA Score Spearman test:\n")
  print(overall_correlation)
})
save_text_output(summary_lines, file.path(output_dir, "analysis-summary.txt"))

message(sprintf("[done] analysis finished successfully; outputs are in %s", output_dir))
