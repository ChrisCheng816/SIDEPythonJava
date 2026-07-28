#!/usr/bin/env Rscript

# Produces Table 2-style ordered-logit results for the Python SIDE metric.
# The two inputs are intentionally kept separate: they represent the model
# trained without SIDE filtering and the model trained after SIDE filtering.

`%||%` <- function(left, right) if (is.null(left)) right else left

parse_args <- function(args) {
  result <- list(no_side = NULL, with_side = NULL, output_dir = NULL)
  index <- 1
  while (index <= length(args)) {
    option <- args[[index]]
    if (option %in% c("--no-side", "--with-side", "--output-dir")) {
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
    omitted = c(omitted_predictors, dropped_constant)
  )
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
default_run_root <- file.path(study_dir, "replay-runs", "2026-04-30-new-side09")
no_side_path <- args$no_side %||% file.path(default_run_root, "evaluation", "500-human-annotation", "no-side_500-human-annotation.csv")
with_side_path <- args$with_side %||% file.path(default_run_root, "evaluation", "500-human-annotation", "with-side-threshold-0_9_500-human-annotation.csv")
output_dir <- args$output_dir %||% file.path(default_run_root, "analysis", "table2-sidepython")

if (!requireNamespace("MASS", quietly = TRUE)) {
  stop("This script requires the R package 'MASS' for polr().", call. = FALSE)
}
for (input_path in c(no_side_path, with_side_path)) {
  if (!file.exists(input_path)) stop(sprintf("Input CSV not found: %s", input_path), call. = FALSE)
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# These are the independent variables selected by the original Table 2 method.
# ROUGE-4-R is absent from the checked-in 500-row extension files, so it is
# reported as omitted unless a future input CSV supplies it.
predictors <- c(
  "BLEU-1", "BERTScore-R", "SentenceBERT_CS", "InferSent_CS", "ROUGE-1-P",
  "ROUGE-4-R", "ROUGE-W-R", "c_coeff", "CodeT5-plus_CS", "SIDEpython"
)
targets <- list(
  "Content Adequacy" = "human_content_adequacy",
  "Conciseness" = "human_conciseness",
  "Fluency" = "human_fluency"
)
conditions <- list(
  "no-side" = no_side_path,
  "with-side" = with_side_path
)
all_results <- list()
summary_lines <- c(
  "Table 2-style ordinal-logit analysis for SIDEpython",
  "Predictors are min-max scaled to [0, 5], matching the original analysis.",
  "The dependent variables are the human 1-5 ratings in the extension data."
)

for (condition_name in names(conditions)) {
  data <- normalize_columns(utils::read.csv(conditions[[condition_name]], check.names = FALSE))
  summary_lines <- c(summary_lines, sprintf("%s input: %s", condition_name, normalizePath(conditions[[condition_name]])))

  for (target_label in names(targets)) {
    result <- fit_ordered_logit(data, target_label, targets[[target_label]], predictors)
    file_stem <- sprintf("%s-%s-polr", condition_name, gsub("[^a-z0-9]+", "-", tolower(target_label)))
    utils::write.csv(result$table, file.path(output_dir, paste0(file_stem, ".csv")), row.names = FALSE)
    write_latex(result$table, file.path(output_dir, paste0(file_stem, ".tex")))

    combined <- result$table
    combined$Condition <- condition_name
    combined$Target <- target_label
    combined <- combined[, c("Condition", "Target", names(result$table)), drop = FALSE]
    all_results[[length(all_results) + 1]] <- combined
    summary_lines <- c(
      summary_lines,
      sprintf("%s / %s: %d complete rows; omitted predictors: %s",
        condition_name,
        target_label,
        result$rows,
        if (length(result$omitted) == 0) "none" else paste(result$omitted, collapse = ", ")
      )
    )
  }
}

utils::write.csv(do.call(rbind, all_results), file.path(output_dir, "table2-sidepython-all-results.csv"), row.names = FALSE)
writeLines(summary_lines, file.path(output_dir, "table2-sidepython-run-summary.txt"))
message(sprintf("[done] SIDEpython ordinal-logit tables written to %s", normalizePath(output_dir)))
