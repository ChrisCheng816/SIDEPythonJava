#!/usr/bin/env Rscript

# Reproducible Python SIDE extension: Table 2 ordered-logit regressions and
# Table 3 PCA.  The input is a human-annotated Python summary CSV with a fresh
# SIDE_score for the exact summaries rated by humans.

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
    "BLEU_1" = "BLEU-1", "ROUGE_1_P" = "ROUGE-1-P", "ROUGE_4_R" = "ROUGE-4-R",
    "ROUGE_W_R" = "ROUGE-W-R", "CodeT5_plus_CS" = "CodeT5-plus_CS",
    "BERTScore_R" = "BERTScore-R", "C_Coeff" = "c_coeff", "SIDE_score" = "SIDEpython"
  )
  names(data) <- unname(ifelse(names(data) %in% names(aliases), aliases[names(data)], names(data)))
  data
}

rouge_4_recall <- function(reference, candidate) {
  tokens <- function(text) {
    values <- unlist(strsplit(tolower(trimws(as.character(text))), "\\s+"))
    values[nzchar(values)]
  }
  reference_tokens <- tokens(reference); candidate_tokens <- tokens(candidate)
  if (length(reference_tokens) < 4 || length(candidate_tokens) < 4) return(0)
  ngrams <- function(values) vapply(seq_len(length(values) - 3), function(i) paste(values[i:(i + 3)], collapse = " "), character(1))
  reference_counts <- table(ngrams(reference_tokens)); candidate_counts <- table(ngrams(candidate_tokens))
  shared <- intersect(names(reference_counts), names(candidate_counts))
  if (length(shared) == 0) return(0)
  sum(pmin(reference_counts[shared], candidate_counts[shared])) / sum(reference_counts)
}

z_score <- function(values) {
  deviation <- stats::sd(values)
  if (!is.finite(deviation) || deviation == 0) stop("Cannot z-score a non-finite or constant predictor.", call. = FALSE)
  (values - mean(values)) / deviation
}

format_number <- function(value) sprintf("%.4f", value)
format_p <- function(value) if (is.na(value)) "--" else if (value < 0.0001) "$<0.0001$" else sprintf("%.4f", value)
latex_escape <- function(value) gsub("_", "\\\\_", value, fixed = TRUE)

write_latex <- function(table_data, path, digits = 4) {
  alignment <- paste0("l", paste(rep("r", ncol(table_data) - 1), collapse = ""))
  lines <- c(sprintf("\\begin{tabular}{%s}", alignment), "\\hline", paste(latex_escape(names(table_data)), collapse = " & "), "\\\\", "\\hline")
  for (index in seq_len(nrow(table_data))) {
    values <- vapply(table_data[index, , drop = FALSE], as.character, character(1))
    lines <- c(lines, paste(latex_escape(values), collapse = " & "), "\\\\")
  }
  writeLines(c(lines, "\\hline", "\\end{tabular}"), path)
}

prepare_model_data <- function(data, target, predictors) {
  raw <- data[, c(target, predictors), drop = FALSE]
  raw[] <- lapply(raw, function(column) suppressWarnings(as.numeric(column)))
  reasons <- list(
    missing_target = sum(!is.finite(raw[[target]])),
    missing_predictor = sum(!apply(raw[, predictors, drop = FALSE], 1, function(row) all(is.finite(row))))
  )
  keep <- is.finite(raw[[target]]) & apply(raw[, predictors, drop = FALSE], 1, function(row) all(is.finite(row)))
  list(data = raw[keep, , drop = FALSE], excluded = sum(!keep), reasons = reasons)
}

fit_polr <- function(model_data, target, predictors, standardized = FALSE) {
  fitted <- model_data
  if (standardized) for (metric in predictors) fitted[[metric]] <- z_score(fitted[[metric]])
  for (metric in predictors) if (length(unique(fitted[[metric]])) < 2) stop(sprintf("Predictor %s is constant.", metric), call. = FALSE)
  fitted[[target]] <- ordered(round(fitted[[target]]))
  response_counts <- table(fitted[[target]])
  if (length(response_counts) < 2) stop(sprintf("%s has fewer than two observed levels.", target), call. = FALSE)
  probabilities <- cumsum(response_counts)[-length(response_counts)] / sum(response_counts)
  start <- c(rep(0, length(predictors)), stats::qlogis(pmin(pmax(probabilities, 1e-6), 1 - 1e-6)))
  formula <- stats::as.formula(paste(sprintf("`%s`", target), "~", paste(sprintf("`%s`", predictors), collapse = " + ")))
  model <- MASS::polr(formula, data = fitted, Hess = TRUE, start = start)
  estimates <- stats::coef(model)[predictors]
  errors <- sqrt(diag(stats::vcov(model)))[predictors]
  z_values <- estimates / errors
  raw_p <- 2 * stats::pnorm(abs(z_values), lower.tail = FALSE)
  data.frame(Metric = predictors, OR = exp(estimates), Value = estimates, `Std. Error` = errors, `Wald z` = z_values, `raw p` = raw_p, check.names = FALSE, row.names = NULL)
}

vif_values <- function(standardized_data, predictors) {
  vapply(predictors, function(metric) {
    others <- setdiff(predictors, metric)
    model <- stats::lm(stats::as.formula(paste(sprintf("`%s`", metric), "~", paste(sprintf("`%s`", others), collapse = " + "))), data = standardized_data)
    1 / (1 - summary(model)$r.squared)
  }, numeric(1))
}

binary_threshold_diagnostics <- function(model_data, target, predictors) {
  response <- round(model_data[[target]])
  levels <- sort(unique(response))
  if (length(levels) < 3) return("Not available: fewer than three observed ordinal levels.")
  thresholds <- levels[-length(levels)]
  coefficients <- lapply(thresholds, function(threshold) {
    binary <- as.integer(response > threshold)
    model <- stats::glm(binary ~ ., data = model_data[, predictors, drop = FALSE], family = stats::binomial())
    stats::coef(model)[predictors]
  })
  values <- do.call(cbind, coefficients); rownames(values) <- predictors; colnames(values) <- paste0(">", thresholds)
  paste(capture.output(print(round(values, 4))), collapse = "\n")
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
study_dir <- normalizePath(file.path(script_dir(), ".."), mustWork = TRUE)
run_root <- file.path(study_dir, "replay-runs", "2026-04-23-base-hf-side09")
input_path <- args$input %||% file.path(run_root, "evaluation", "metrics", "table2-sidepython", "human-annotation_with_fresh_side.csv")
output_dir <- args$output_dir %||% file.path(run_root, "evaluation", "metrics", "table2-sidepython")
if (!requireNamespace("MASS", quietly = TRUE)) stop("This script requires MASS.", call. = FALSE)
if (!file.exists(input_path)) stop(sprintf("Input CSV not found: %s", input_path), call. = FALSE)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

all_predictors <- c("BLEU-1", "BERTScore-R", "SentenceBERT_CS", "InferSent_CS", "ROUGE-1-P", "ROUGE-4-R", "ROUGE-W-R", "c_coeff", "CodeT5-plus_CS", "SIDEpython")
predictors <- setdiff(all_predictors, "ROUGE-4-R")
targets <- c("Content Adequacy" = "human_content_adequacy", "Conciseness" = "human_conciseness", "Fluency" = "human_fluency")
data <- normalize_columns(utils::read.csv(input_path, check.names = FALSE))
if (!("ROUGE-4-R" %in% names(data))) {
  if (!all(c("originalComment", "codeComment") %in% names(data))) stop("Cannot compute ROUGE-4-R without originalComment and codeComment.", call. = FALSE)
  data[["ROUGE-4-R"]] <- mapply(rouge_4_recall, data[["originalComment"]], data[["codeComment"]])
}
required <- c(all_predictors, unname(targets))
missing <- setdiff(required, names(data))
if (length(missing) > 0) stop(sprintf("Input is missing: %s", paste(missing, collapse = ", ")), call. = FALSE)

diagnostics <- c("# Python SIDE ordered-logit diagnostics", sprintf("Input: %s", normalizePath(input_path)), sprintf("Rows read: %d", nrow(data)), "", "## ROUGE-4-R diagnostics before exclusion")
r4 <- as.numeric(data[["ROUGE-4-R"]])
diagnostics <- c(diagnostics, sprintf("zero proportion: %.6f", mean(r4 == 0, na.rm = TRUE)), sprintf("mean: %.6f", mean(r4, na.rm = TRUE)), sprintf("sd (ddof=1): %.6f", stats::sd(r4, na.rm = TRUE)), sprintf("non-zero samples: %d", sum(r4 > 0, na.rm = TRUE)))
for (label in names(targets)) {
  target <- targets[[label]]
  diagnostics <- c(diagnostics, "", sprintf("### ROUGE-4-R == 0 versus > 0 by %s", label), capture.output(print(table(round(data[[target]]), ifelse(r4 > 0, ">0", "==0")))))
}

# Required pre-exclusion Fluency estimate.  Any failure is documented rather
# than hidden; the nine-predictor model is still the specified final analysis.
pre_fluency <- tryCatch({
  prepared <- prepare_model_data(data, targets[["Fluency"]], all_predictors)
  fitted <- fit_polr(prepared$data, targets[["Fluency"]], all_predictors, standardized = FALSE)
  fitted[fitted$Metric == "ROUGE-4-R", c("Value", "Std. Error", "OR", "raw p")]
}, error = function(error) paste("pre-exclusion Fluency model failed:", conditionMessage(error)))
diagnostics <- c(diagnostics, "", "### Pre-exclusion Fluency ROUGE-4-R estimate", capture.output(print(pre_fluency)))

multi_results <- list(); uni_results <- list()
for (label in names(targets)) {
  target <- targets[[label]]
  prepared <- prepare_model_data(data, target, predictors)
  model_data <- prepared$data
  raw_fit <- fit_polr(model_data, target, predictors, standardized = FALSE)
  standardized_fit <- fit_polr(model_data, target, predictors, standardized = TRUE)
  p_difference <- max(abs(raw_fit$`raw p` - standardized_fit$`raw p`), na.rm = TRUE)
  diagnostics <- c(diagnostics, "", sprintf("## %s", label), sprintf("rows entering model: %d", nrow(model_data)), sprintf("rows excluded: %d", prepared$excluded), sprintf("excluded: non-finite target=%d; non-finite predictor row=%d", prepared$reasons$missing_target, prepared$reasons$missing_predictor), "rating counts and proportions:", capture.output(print(data.frame(count = table(round(model_data[[target]])), proportion = prop.table(table(round(model_data[[target]]))))), row.names = FALSE), sprintf("raw-versus-z-score max p difference: %.12f", p_difference))
  if (!is.finite(p_difference) || p_difference > 1e-6) {
    writeLines(diagnostics, file.path(output_dir, "sidepython-regression-pca-diagnostics.md"))
    stop(sprintf("Wald p-value invariance failed for %s (difference %.12f). Diagnostics were written before stopping.", label, p_difference), call. = FALSE)
  }
  low_levels <- names(table(round(model_data[[target]])))[table(round(model_data[[target]])) < 10]
  diagnostics <- c(diagnostics, if (length(low_levels) == 0) "rating levels below 10: none" else paste("rating levels below 10:", paste(low_levels, collapse = ", ")))
  z_data <- model_data; for (metric in predictors) z_data[[metric]] <- z_score(z_data[[metric]])
  diagnostics <- c(diagnostics, "VIF:", capture.output(print(round(vif_values(z_data, predictors), 4))), sprintf("design-matrix condition number: %.4f", kappa(stats::model.matrix(~ 0 + ., data = z_data[, predictors, drop = FALSE]))), "proportional-odds alternative diagnostic (binary-logit coefficients by threshold):", binary_threshold_diagnostics(z_data, target, predictors))
  standardized_fit$`BH p` <- stats::p.adjust(standardized_fit$`raw p`, method = "BH")
  multi_results[[label]] <- standardized_fit
  univariate <- do.call(rbind, lapply(predictors, function(metric) fit_polr(model_data, target, metric, standardized = TRUE)))
  univariate$`BH p` <- stats::p.adjust(univariate$`raw p`, method = "BH")
  uni_results[[label]] <- univariate
}

regression_table <- data.frame(Metric = predictors, check.names = FALSE)
for (label in names(targets)) {
  multi <- multi_results[[label]][match(predictors, multi_results[[label]]$Metric), ]
  uni <- uni_results[[label]][match(predictors, uni_results[[label]]$Metric), ]
  regression_table[[paste(label, "Multi OR")]] <- vapply(multi$OR, format_number, character(1))
  regression_table[[paste(label, "Multi p")]] <- vapply(multi$`raw p`, format_p, character(1))
  regression_table[[paste(label, "Multi BH p")]] <- vapply(multi$`BH p`, format_p, character(1))
  regression_table[[paste(label, "Uni OR")]] <- vapply(uni$OR, format_number, character(1))
  regression_table[[paste(label, "Uni p")]] <- vapply(uni$`raw p`, format_p, character(1))
  regression_table[[paste(label, "Uni BH p")]] <- vapply(uni$`BH p`, format_p, character(1))
}

pca_prepared <- prepare_model_data(data, targets[["Content Adequacy"]], predictors)
pca_data <- pca_prepared$data[, predictors, drop = FALSE]
for (metric in predictors) pca_data[[metric]] <- z_score(pca_data[[metric]])
pca <- stats::prcomp(pca_data, scale. = FALSE)
importance <- summary(pca)$importance
pca_table <- rbind(importance["Proportion of Variance", , drop = FALSE], importance["Cumulative Proportion", , drop = FALSE], pca$rotation)
pca_table <- data.frame(Metric = rownames(pca_table), pca_table, check.names = FALSE, row.names = NULL)
names(pca_table)[-1] <- paste0("PC", seq_len(ncol(pca$rotation)))
pca_table[-1] <- lapply(pca_table[-1], function(column) sprintf("%.3f", as.numeric(column)))
side_row <- pca$rotation["SIDEpython", ]
side_component <- which.max(abs(side_row))
diagnostics <- c(diagnostics, "", "## PCA", sprintf("rows entering PCA: %d", nrow(pca_data)), sprintf("SIDEpython's largest absolute loading: PC%d = %.6f; variance proportion = %.6f", side_component, side_row[[side_component]], importance["Proportion of Variance", side_component]))

utils::write.csv(regression_table, file.path(output_dir, "table2-sidepython-regression.csv"), row.names = FALSE)
write_latex(regression_table, file.path(output_dir, "table2-sidepython-regression.tex"))
utils::write.csv(pca_table, file.path(output_dir, "table3-sidepython-pca.csv"), row.names = FALSE)
write_latex(pca_table, file.path(output_dir, "table3-sidepython-pca.tex"))
writeLines(diagnostics, file.path(output_dir, "sidepython-regression-pca-diagnostics.md"))
message(sprintf("[done] diagnostics and Table 2/Table 3 fragments written to %s", normalizePath(output_dir)))
