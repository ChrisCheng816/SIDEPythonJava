#!/usr/bin/env Rscript

# Produces the publication Table 2 ordered-logit results for SIDEpython.
# The input must contain human ratings for the same code summaries scored by
# SIDEpython; it must not be a CSV whose codeComment values were replaced by a
# later model's predictions.

`%||%` <- function(left, right) if (is.null(left)) right else left

parse_args <- function(args) {
  result <- list(input = NULL, output_dir = NULL)
  index <- 1
  while (index <= length(args)) {
    option <- args[[index]]
    if (option %in% c("--input", "--output-dir")) {
      if (index == length(args)) stop(sprintf("Missing value for %s.", option), call. = FALSE)
      key <- sub("^--", "", gsub("-", "_", option))
      result[[key]] <- args[[index + 1]]
      index <- index + 2
    } else {
      stop(sprintf("Unknown option: %s", option), call. = FALSE)
    }
  }
  result
}

script_dir <- function() {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) == 0) return(normalizePath(getwd()))
  dirname(normalizePath(sub("^--file=", "", file_arg[[1]])))
}

normalize_columns <- function(data) {
  aliases <- c(
    "BLEU_1" = "BLEU-1",
    "ROUGE_1_P" = "ROUGE-1-P",
    "ROUGE-4-R" = "ROUGE-4-R",
    "ROUGE_4_R" = "ROUGE-4-R",
    "ROUGE_W_R" = "ROUGE-W-R",
    "CodeT5_plus_CS" = "CodeT5-plus_CS",
    "BERTScore_R" = "BERTScore-R",
    "SentenceBERT_CS" = "SentenceBERT_CS",
    "InferSent_CS" = "InferSent_CS",
    "C_Coeff" = "c_coeff",
    "SIDE_score" = "SIDEpython"
  )
  names(data) <- unname(ifelse(names(data) %in% names(aliases), aliases[names(data)], names(data)))
  data
}

rouge_n_recall <- function(reference, candidate, n = 4L) {
  tokens <- function(text) {
    values <- unlist(strsplit(tolower(trimws(as.character(text))), "\\s+"))
    values[nzchar(values)]
  }
  ref_tokens <- tokens(reference)
  candidate_tokens <- tokens(candidate)
  if (length(ref_tokens) < n || length(candidate_tokens) < n) return(0)
  ngrams <- function(values) {
    vapply(seq_len(length(values) - n + 1L), function(index) {
      paste(values[index:(index + n - 1L)], collapse = " ")
    }, character(1))
  }
  ref_counts <- table(ngrams(ref_tokens))
  candidate_counts <- table(ngrams(candidate_tokens))
  shared <- intersect(names(ref_counts), names(candidate_counts))
  if (length(shared) == 0) return(0)
  sum(pmin(ref_counts[shared], candidate_counts[shared])) / sum(ref_counts)
}

ensure_rouge_4_recall <- function(data) {
  if ("ROUGE-4-R" %in% names(data)) return(data)
  required <- c("originalComment", "codeComment")
  missing <- setdiff(required, names(data))
  if (length(missing) > 0) {
    stop(sprintf("Cannot compute ROUGE-4-R; input is missing: %s", paste(missing, collapse = ", ")), call. = FALSE)
  }
  message("[info] ROUGE-4-R is absent; computing 4-gram recall from originalComment and codeComment.")
  data[["ROUGE-4-R"]] <- mapply(rouge_n_recall, data[["originalComment"]], data[["codeComment"]])
  data
}

normalize_java_columns <- function(data) {
  aliases <- c(
    "jaccard" = "Jaccard", "bleu-A" = "BLEU-A", "bleu-1" = "BLEU-1", "bleu-2" = "BLEU-2",
    "bleu-3" = "BLEU-3", "bleu-4" = "BLEU-4", "tfidf_cosine" = "TF_IDF_CS",
    "USE_cosine_similarity" = "USE_CS", "bert-score-precision" = "BERTScore-P",
    "bert-score-recall" = "BERTScore-R", "bert-score-f1" = "BERTScore-F1",
    "sentence_bert_cosine_similarity" = "SentenceBERT_CS",
    "infersent_cosine_similarity" = "InferSent_CS", "meteor" = "METEOR",
    "rouge-l-f1" = "ROUGE-L-F1", "rouge-1-f1" = "ROUGE-1-F1", "rouge-1-p" = "ROUGE-1-P",
    "rouge-1-r" = "ROUGE-1-R", "rouge-2-f1" = "ROUGE-2-F1", "rouge-2-p" = "ROUGE-2-P",
    "rouge-2-r" = "ROUGE-2-R", "rouge-3-f1" = "ROUGE-3-F1", "rouge-3-p" = "ROUGE-3-P",
    "rouge-3-r" = "ROUGE-3-R", "rouge-4-f1" = "ROUGE-4-F1", "rouge-4-p" = "ROUGE-4-P",
    "rouge-4-r" = "ROUGE-4-R", "rouge-l-p" = "ROUGE-L-P", "rouge-l-r" = "ROUGE-L-R",
    "rouge-w-f1" = "ROUGE-W-F1", "rouge-w-p" = "ROUGE-W-P", "rouge-w-r" = "ROUGE-W-R",
    "tfidf_euclidean" = "TF_IDF_ED", "chrf_score" = "chrF",
    "USE_euclidean_distance" = "USE_ED", "sentence_bert_euclidean_distance" = "SentenceBERT_ED",
    "infersent_euclidean_distance" = "InferSent_ED", "CodeT5-plus-cosine-similarity" = "CodeT5-plus_CS"
  )
  names(data) <- unname(ifelse(names(data) %in% names(aliases), aliases[names(data)], names(data)))
  data
}

scale_to_five <- function(values) {
  lower <- min(values, na.rm = TRUE)
  upper <- max(values, na.rm = TRUE)
  if (isTRUE(all.equal(lower, upper))) return(rep(NA_real_, length(values)))
  (values - lower) * 5 / (upper - lower)
}

fit_ordered_logit <- function(data, target_label, target_column, requested_predictors) {
  available_predictors <- intersect(requested_predictors, names(data))
  omitted_predictors <- setdiff(requested_predictors, available_predictors)
  if (length(available_predictors) == 0) {
    stop(sprintf("No requested predictors are available for %s.", target_label), call. = FALSE)
  }
  if (!(target_column %in% names(data))) {
    stop(sprintf("Missing target column '%s' for %s.", target_column, target_label), call. = FALSE)
  }

  model_data <- data[, c(target_column, available_predictors), drop = FALSE]
  model_data[] <- lapply(model_data, function(column) suppressWarnings(as.numeric(column)))
  model_data <- stats::na.omit(model_data)
  model_data <- model_data[apply(model_data, 1, function(row) all(is.finite(row))), , drop = FALSE]
  if (nrow(model_data) == 0) {
    stop(sprintf("No complete cases are available for %s.", target_label), call. = FALSE)
  }

  non_constant <- vapply(
    available_predictors,
    function(metric) !isTRUE(all.equal(min(model_data[[metric]]), max(model_data[[metric]]))),
    logical(1)
  )
  dropped_constant <- available_predictors[!non_constant]
  available_predictors <- available_predictors[non_constant]
  if (length(available_predictors) == 0) {
    stop(sprintf("All available predictors are constant for %s.", target_label), call. = FALSE)
  }

  for (metric in available_predictors) {
    model_data[[metric]] <- scale_to_five(model_data[[metric]])
  }
  # polr cannot initialize a rank-deficient design matrix. Keep the first
  # independent predictors in the declared Table 2 order and report any that
  # are exact linear dependencies as omitted controls.
  safe_predictor_names <- paste0("metric_", seq_along(available_predictors))
  design_data <- model_data[, available_predictors, drop = FALSE]
  names(design_data) <- safe_predictor_names
  design <- stats::model.matrix(~ ., data = design_data)
  design_rank <- qr(design)$rank
  independent_columns <- qr(design)$pivot[seq_len(design_rank)]
  independent_safe_names <- setdiff(colnames(design)[independent_columns], "(Intercept)")
  independent_predictors <- available_predictors[match(independent_safe_names, safe_predictor_names)]
  dropped_dependent <- setdiff(available_predictors, independent_predictors)
  available_predictors <- independent_predictors
  if (length(available_predictors) == 0) {
    stop(sprintf("All predictors are linearly dependent for %s.", target_label), call. = FALSE)
  }
  model_data[[target_column]] <- ordered(round(model_data[[target_column]]))
  if (length(levels(model_data[[target_column]])) < 2) {
    stop(sprintf("%s has fewer than two observed ordinal levels.", target_label), call. = FALSE)
  }

  formula <- stats::as.formula(paste(
    sprintf("`%s`", target_column),
    "~",
    paste(sprintf("`%s`", available_predictors), collapse = " + ")
  ))
  model <- MASS::polr(formula, data = model_data, Hess = TRUE)
  coefficients <- coef(summary(model))[seq_along(available_predictors), , drop = FALSE]
  p_values <- 2 * stats::pnorm(abs(coefficients[, "t value"]), lower.tail = FALSE)
  table <- data.frame(
    Metric = rownames(coefficients),
    OR = exp(coefficients[, "Value"]),
    Value = coefficients[, "Value"],
    `Std. Error` = coefficients[, "Std. Error"],
    `t value` = coefficients[, "t value"],
    `p value` = p_values,
    check.names = FALSE,
    row.names = NULL
  )
  list(
    table = table,
    rows = nrow(model_data),
    omitted = c(omitted_predictors, dropped_constant, dropped_dependent)
  )
}

legacy_java_core_metrics <- function() {
  metrics <- c("BLEU-1", "BERTScore-R", "SentenceBERT_CS", "InferSent_CS", "ROUGE-1-P", "ROUGE-4-R", "ROUGE-W-R", "c_coeff", "CodeT5-plus_CS", "SIDE-Java")
  dimensions <- c("Overall DA Score", "Content Adequacy", "Conciseness", "Fluency")
  odds_ratios <- rbind(
    c(.9990, 1.0029, 1.0057, .9980, 1.0058, 1.0024, .9996, 1.0143, 1.0044, 1.0205),
    c(.9672, 1.0525, 1.1121, 1.0000, 1.0943, .9914, 1.0437, 1.3729, 1.1413, 1.6265),
    c(1.0656, 1.0160, 1.0587, 1.0079, 1.0075, .9937, 1.0954, 1.2433, 1.0630, 1.3844),
    c(1.0650, 1.0559, .9958, 1.0543, 1.0334, 1.0304, .9672, 1.2080, 1.0249, 1.2826)
  )
  p_values <- rbind(
    c(.6222, .2943, .0100, .5250, .0033, .04667, .8040, 0, .0220, 0),
    c(.4200, .3929, .0225, 1, .0260, .7811, .3700, 0, 0, 0),
    c(.1575, .8770, .2617, .8770, .8770, .8770, .0267, 0, .1700, 0),
    c(.2267, .4600, .9170, .4600, .4600, .4600, .4600, 0, .5444, 0)
  )
  do.call(rbind, lapply(seq_along(dimensions), function(index) data.frame(
    Variant = "Original SIDE-Java", Condition = "not-applicable", Dimension = dimensions[[index]], Metric = metrics,
    OR = odds_ratios[index, ], Value = NA_real_, `Std. Error` = NA_real_, `t value` = NA_real_, `p value` = p_values[index, ], check.names = FALSE
  )))
}

reproduce_java_core_metrics <- function(input_path) {
  metrics <- c(
    "Jaccard", "BLEU-A", "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "TF_IDF_CS", "USE_CS",
    "BERTScore-P", "BERTScore-R", "BERTScore-F1", "SentenceBERT_CS", "InferSent_CS", "METEOR",
    "ROUGE-L-F1", "c_coeff", "ROUGE-1-F1", "ROUGE-1-P", "ROUGE-1-R", "ROUGE-2-F1",
    "ROUGE-2-P", "ROUGE-2-R", "ROUGE-3-F1", "ROUGE-3-P", "ROUGE-3-R", "ROUGE-4-F1",
    "ROUGE-4-P", "ROUGE-4-R", "ROUGE-L-P", "ROUGE-L-R", "ROUGE-W-F1", "ROUGE-W-P",
    "ROUGE-W-R", "TF_IDF_ED", "chrF", "USE_ED", "SentenceBERT_ED", "InferSent_ED",
    "CodeT5-plus_CS", "SIDE"
  )
  data <- normalize_java_columns(utils::read.csv(input_path, check.names = FALSE))
  required <- c("mid", metrics, "Overall DA Score", "Content Adequacy", "Conciseness", "Fluency")
  missing <- setdiff(required, names(data))
  if (length(missing) > 0) stop(sprintf("Java input is missing: %s", paste(missing, collapse = ", ")), call. = FALSE)
  data <- stats::na.omit(data[data$mid != 0, c(metrics, "Overall DA Score", "Content Adequacy", "Conciseness", "Fluency"), drop = FALSE])
  # redun() is sensitive to metric names such as "BLEU-1" and "Overall DA Score".
  # Select with temporary safe names, then map the retained metrics back.
  safe_metric_names <- paste0("metric_", seq_along(metrics))
  selection_data <- data[, metrics, drop = FALSE]
  names(selection_data) <- safe_metric_names
  formula <- stats::as.formula(paste("~", paste(safe_metric_names, collapse = " + ")))
  selected_safe_names <- Hmisc::redun(formula, data = selection_data, r2 = 0.8, nk = 0)$In
  reduced <- metrics[match(selected_safe_names[!is.na(selected_safe_names)], safe_metric_names)]

  targets <- list("Overall DA Score" = 100, "Content Adequacy" = 5, "Conciseness" = 5, "Fluency" = 5)
  core_metrics <- reduced
  rows <- lapply(names(targets), function(target) {
    model_data <- data
    for (metric in reduced) {
      model_data[[metric]] <- (model_data[[metric]] - min(model_data[[metric]])) * targets[[target]] /
        (max(model_data[[metric]]) - min(model_data[[metric]]))
    }
    model_data[[target]] <- ordered(round(model_data[[target]]))
    model <- MASS::polr(
      stats::as.formula(paste(sprintf("`%s`", target), "~", paste(sprintf("`%s`", reduced), collapse = " + "))),
      data = model_data,
      Hess = TRUE
    )
    coefficients <- coef(summary(model))
    do.call(rbind, lapply(core_metrics, function(metric) {
      row_name <- c(metric, paste0("`", metric, "`"))
      row_name <- row_name[row_name %in% rownames(coefficients)][1]
      if (is.na(row_name)) stop(sprintf("%s was not retained in the reproduced Java regression.", metric), call. = FALSE)
      coefficient <- coefficients[row_name, , drop = FALSE]
      data.frame(
        Variant = "Reproduced SIDE-Java",
        Condition = "not-applicable",
        Dimension = target,
        Metric = if (metric == "SIDE") "SIDE-Java" else metric,
        OR = exp(coefficient[1, "Value"]),
        Value = coefficient[1, "Value"],
        `Std. Error` = coefficient[1, "Std. Error"],
        `t value` = coefficient[1, "t value"],
        `p value` = 2 * stats::pnorm(abs(coefficient[1, "t value"]), lower.tail = FALSE),
        check.names = FALSE
      )
    }))
  })
  do.call(rbind, rows)
}

write_latex <- function(table_data, output_path) {
  columns <- names(table_data)
  alignment <- paste0("l", paste(rep("r", length(columns) - 1), collapse = ""))
  escape <- function(value) gsub("_", "\\_", value, fixed = TRUE)
  lines <- c(
    sprintf("\\begin{tabular}{%s}", alignment),
    "\\hline", paste(escape(columns), collapse = " & "), "\\\\", "\\hline"
  )
  for (row in seq_len(nrow(table_data))) {
    values <- vapply(table_data[row, , drop = FALSE], as.character, character(1))
    lines <- c(lines, paste(escape(values), collapse = " & "), "\\\\")
  }
  writeLines(c(lines, "\\hline", "\\end{tabular}"), output_path)
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
scripts_dir <- script_dir()
study_dir <- normalizePath(file.path(scripts_dir, ".."), mustWork = TRUE)
default_run_root <- file.path(study_dir, "replay-runs", "2026-04-23-base-hf-side09")
input_path <- args$input %||% file.path(default_run_root, "evaluation", "metrics", "table2-sidepython", "human-annotation_with_fresh_side.csv")
output_dir <- args$output_dir %||% file.path(default_run_root, "evaluation", "metrics", "table2-sidepython")

if (!requireNamespace("MASS", quietly = TRUE)) {
  stop("This script requires the R package 'MASS' for polr().", call. = FALSE)
}
if (!file.exists(input_path)) stop(sprintf("Input CSV not found: %s", input_path), call. = FALSE)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Match the paper's Table 2: SIDEpython is interpreted while controlling for
# the other selected automatic metrics, all recomputed from the same Python
# summaries in the input CSV.
predictors <- c(
  "BLEU-1", "BERTScore-R", "SentenceBERT_CS", "InferSent_CS", "ROUGE-1-P",
  "ROUGE-4-R", "ROUGE-W-R", "c_coeff", "CodeT5-plus_CS", "SIDEpython"
)
targets <- list(
  "Content Adequacy" = "human_content_adequacy",
  "Conciseness" = "human_conciseness",
  "Fluency" = "human_fluency"
)
summary_lines <- c(
  "Table 2 ordered-logit analysis for SIDEpython",
  sprintf("Input: %s", normalizePath(input_path)),
  "Responses: human content adequacy, conciseness, and fluency (ordinal 1-5).",
  "All predictors are min-max scaled to [0, 5], matching each response scale."
)

data <- ensure_rouge_4_recall(normalize_columns(utils::read.csv(input_path, check.names = FALSE)))
missing <- setdiff(c("codeComment", "SIDEpython", unname(unlist(targets))), names(data))
if (length(missing) > 0) {
  stop(sprintf("Input is missing: %s", paste(missing, collapse = ", ")), call. = FALSE)
}

all_results <- lapply(names(targets), function(target_label) {
  result <- fit_ordered_logit(data, target_label, targets[[target_label]], predictors)
  summary_lines <<- c(
    summary_lines,
    sprintf("%s: %d complete rows; omitted predictors: %s",
      target_label,
      result$rows,
      if (length(result$omitted) == 0) "none" else paste(result$omitted, collapse = ", ")
    )
  )
  data.frame(
    Variant = "SIDEpython",
    Condition = "human-annotated",
    Dimension = target_label,
    result$table,
    check.names = FALSE
  )
})
combined_results <- do.call(rbind, all_results)

format_cell <- function(or_value, p_value) {
  p_text <- if (is.na(p_value)) "--" else if (p_value < 0.0001) "$p<0.0001$" else sprintf("$p=%.4f$", p_value)
  sprintf("%.4f (%s)", or_value, p_text)
}

dimensions <- names(targets)
combined_results$MetricLabel <- ifelse(
  combined_results$Condition == "not-applicable",
  paste(combined_results$Variant, combined_results$Metric, sep = ": "),
  paste(combined_results$Variant, combined_results$Condition, combined_results$Metric, sep = ": ")
)
metric_labels <- unique(combined_results$MetricLabel)
wide_results <- data.frame(Metric = metric_labels, check.names = FALSE)
for (dimension in dimensions) {
  values <- vapply(metric_labels, function(label) {
    row <- combined_results[combined_results$MetricLabel == label & combined_results$Dimension == dimension, , drop = FALSE]
    if (nrow(row) == 0) return("--")
    format_cell(row$OR[[1]], row$`p value`[[1]])
  }, character(1))
  wide_results[[dimension]] <- values
}
utils::write.csv(wide_results, file.path(output_dir, "table2-regression-summary.csv"), row.names = FALSE)
write_latex(wide_results, file.path(output_dir, "table2-regression-summary.tex"))
writeLines(summary_lines, file.path(output_dir, "table2-sidepython-run-summary.txt"))
message(sprintf("[done] SIDEpython Table 2 written to %s", normalizePath(output_dir)))
