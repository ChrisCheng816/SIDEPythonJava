#!/usr/bin/env Rscript

# Python SIDE extension: Table 2 ordered-logit results and Table 3 PCA.
# Table 2 follows the Java SIDE Table 2 procedure: min-max scaling to the
# response range, MASS::polr(), and unadjusted Wald p-values. ROUGE-4-R is
# deliberately excluded from every model because of its sparse distribution.
# The Python annotation file has no separately collected Overall DA label.
# Its Overall column below is therefore explicitly a composite: the mean of
# the three observed human ratings, not a replacement name for Overall DA.

`%||%` <- function(left, right) if (is.null(left)) right else left

parse_args <- function(args) {
  result <- list(input = NULL, output_dir = NULL)
  index <- 1
  while (index <= length(args)) {
    option <- args[[index]]
    if (!(option %in% c("--input", "--output-dir"))) stop(sprintf("Unknown option: %s", option), call. = FALSE)
    if (index == length(args)) stop(sprintf("Missing value for %s.", option), call. = FALSE)
    result[[sub("^--", "", gsub("-", "_", option))]] <- args[[index + 1]]
    index <- index + 2
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
    "BLEU_1" = "BLEU-1", "ROUGE_1_P" = "ROUGE-1-P", "ROUGE_W_R" = "ROUGE-W-R",
    "CodeT5_plus_CS" = "CodeT5-plus_CS", "BERTScore_R" = "BERTScore-R",
    "C_Coeff" = "c_coeff", "SIDE_score" = "SIDEpython"
  )
  names(data) <- unname(ifelse(names(data) %in% names(aliases), aliases[names(data)], names(data)))
  data
}

scale_to_range <- function(values, upper_bound) {
  lower <- min(values)
  upper <- max(values)
  if (!is.finite(lower) || !is.finite(upper) || lower == upper) stop("A Table 2 predictor is constant or non-finite.", call. = FALSE)
  (values - lower) * upper_bound / (upper - lower)
}

format_p <- function(value) if (value < 0.0001) "$<0.0001$" else sprintf("$p=%.4f$", value)
format_cell <- function(or_value, p_value) sprintf("%.4f (%s)", or_value, format_p(p_value))
latex_escape <- function(value) gsub("_", "\\\\_", value, fixed = TRUE)

write_latex <- function(table_data, path) {
  alignment <- paste0("l", paste(rep("r", ncol(table_data) - 1), collapse = ""))
  lines <- c(sprintf("\\begin{tabular}{%s}", alignment), "\\hline", paste(latex_escape(names(table_data)), collapse = " & "), "\\\\", "\\hline")
  for (index in seq_len(nrow(table_data))) {
    values <- vapply(table_data[index, , drop = FALSE], as.character, character(1))
    lines <- c(lines, paste(latex_escape(values), collapse = " & "), "\\\\")
  }
  writeLines(c(lines, "\\hline", "\\end{tabular}"), path)
}

fit_ordered_logit <- function(data, target, predictors) {
  model_data <- data[, c(target, predictors), drop = FALSE]
  model_data[] <- lapply(model_data, function(column) suppressWarnings(as.numeric(column)))
  complete <- stats::complete.cases(model_data) & apply(model_data, 1, function(row) all(is.finite(row)))
  model_data <- model_data[complete, , drop = FALSE]
  if (nrow(model_data) == 0) stop(sprintf("No complete rows for %s.", target), call. = FALSE)
  for (metric in predictors) model_data[[metric]] <- scale_to_range(model_data[[metric]], 5)
  model_data[[target]] <- ordered(round(model_data[[target]]))
  counts <- table(model_data[[target]])
  if (length(counts) < 2) stop(sprintf("%s has fewer than two rating levels.", target), call. = FALSE)
  formula <- stats::as.formula(paste(sprintf("`%s`", target), "~", paste(sprintf("`%s`", predictors), collapse = " + ")))
  model <- MASS::polr(formula, data = model_data, Hess = TRUE)
  # polr preserves backticks in non-syntactic coefficient names (e.g.
  # `BLEU-1`), so name indexing by the human-facing metric label returns NA.
  # The first coefficients are in the formula's predictor order.
  estimates <- unname(stats::coef(model)[seq_along(predictors)])
  errors <- unname(sqrt(diag(stats::vcov(model)))[seq_along(predictors)])
  names(estimates) <- predictors
  names(errors) <- predictors
  z_values <- estimates / errors
  raw_p <- 2 * stats::pnorm(abs(z_values), lower.tail = FALSE)
  data.frame(Metric = predictors, OR = exp(estimates), `p value` = raw_p, check.names = FALSE, row.names = NULL)
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
study_dir <- normalizePath(file.path(script_dir(), ".."), mustWork = TRUE)
run_root <- file.path(study_dir, "replay-runs", "2026-04-23-base-hf-side09")
input_path <- args$input %||% file.path(run_root, "evaluation", "metrics", "table2-sidepython", "human-annotation_with_fresh_side.csv")
output_dir <- args$output_dir %||% file.path(run_root, "evaluation", "metrics", "table2-sidepython")
if (!requireNamespace("MASS", quietly = TRUE)) stop("This script requires MASS.", call. = FALSE)
if (!file.exists(input_path)) stop(sprintf("Input CSV not found: %s", input_path), call. = FALSE)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

predictors <- c("BLEU-1", "BERTScore-R", "SentenceBERT_CS", "InferSent_CS", "ROUGE-1-P", "ROUGE-W-R", "c_coeff", "CodeT5-plus_CS", "SIDEpython")
human_dimensions <- c("human_content_adequacy", "human_conciseness", "human_fluency")
targets <- c(
  "Overall (mean of three human ratings)" = "overall_mean_human_rating",
  "Content Adequacy" = "human_content_adequacy",
  "Conciseness" = "human_conciseness",
  "Fluency" = "human_fluency"
)
data <- normalize_columns(utils::read.csv(input_path, check.names = FALSE))
missing <- setdiff(c(predictors, human_dimensions), names(data))
if (length(missing) > 0) stop(sprintf("Input is missing: %s", paste(missing, collapse = ", ")), call. = FALSE)
# Keep a missing component missing, so fit_ordered_logit reports it through its
# complete-case handling instead of silently averaging the remaining ratings.
human_matrix <- data[, human_dimensions, drop = FALSE]
human_matrix[] <- lapply(human_matrix, function(column) suppressWarnings(as.numeric(column)))
data[["overall_mean_human_rating"]] <- rowMeans(human_matrix, na.rm = FALSE)

fits <- lapply(targets, function(target) fit_ordered_logit(data, target, predictors))
table2 <- data.frame(Metric = predictors, check.names = FALSE)
for (label in names(targets)) {
  fit <- fits[[label]][match(predictors, fits[[label]]$Metric), ]
  table2[[label]] <- mapply(format_cell, fit$OR, fit$`p value`, USE.NAMES = FALSE)
}

# Table 3: PCA on exactly the same nine predictors, matching Java Table 1's
# unscaled PCA step.
pca_data <- data[, predictors, drop = FALSE]
pca_data[] <- lapply(pca_data, function(column) suppressWarnings(as.numeric(column)))
pca_data <- pca_data[stats::complete.cases(pca_data) & apply(pca_data, 1, function(row) all(is.finite(row))), , drop = FALSE]
if (nrow(pca_data) == 0) stop("No complete finite rows for Table 3 PCA.", call. = FALSE)
pca <- stats::prcomp(pca_data, scale. = FALSE)
importance <- summary(pca)$importance
table3 <- rbind(importance["Proportion of Variance", , drop = FALSE], importance["Cumulative Proportion", , drop = FALSE], pca$rotation)
table3 <- data.frame(Metric = rownames(table3), table3, check.names = FALSE, row.names = NULL)
names(table3)[-1] <- paste0("PC", seq_len(ncol(pca$rotation)))
table3[-1] <- lapply(table3[-1], function(column) sprintf("%.3f", as.numeric(column)))
side_loading <- pca$rotation["SIDEpython", ]
side_component <- which.max(abs(side_loading))

utils::write.csv(table2, file.path(output_dir, "table2-sidepython-regression.csv"), row.names = FALSE)
write_latex(table2, file.path(output_dir, "table2-sidepython-regression.tex"))
utils::write.csv(table3, file.path(output_dir, "table3-sidepython-pca.csv"), row.names = FALSE)
write_latex(table3, file.path(output_dir, "table3-sidepython-pca.tex"))
writeLines(c(
  "SIDEpython Table 2 / Table 3 summary",
  sprintf("Input: %s", normalizePath(input_path)),
  "Table 2: nine predictors min-max scaled to [0, 5]; ROUGE-4-R excluded from every model; p-values are unadjusted Wald p-values.",
  "Overall column: mean of Content Adequacy, Conciseness, and Fluency human ratings; this dataset has no separately collected Overall DA Score.",
  sprintf("Table 3 PCA rows: %d", nrow(pca_data)),
  sprintf("SIDEpython largest absolute loading: PC%d = %.6f; variance proportion = %.6f", side_component, side_loading[[side_component]], importance["Proportion of Variance", side_component])
), file.path(output_dir, "table2-table3-summary.txt"))
message(sprintf("[done] Table 2 and Table 3 written to %s", normalizePath(output_dir)))
