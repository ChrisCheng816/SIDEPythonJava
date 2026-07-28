#!/usr/bin/env Rscript

# Reproduces Table 1 (PCA) from the SIDE paper.  The legacy table is kept as
# a fixed notebook reference so a fresh run can be compared directly to it.

parse_args <- function(args) {
  result <- list(
    input = NULL,
    output_dir = NULL
  )

  index <- 1
  while (index <= length(args)) {
    option <- args[[index]]
    if (option %in% c("--input", "--output-dir")) {
      if (index == length(args)) {
        stop(sprintf("Missing value for %s.", option), call. = FALSE)
      }
      result[[sub("^--", "", gsub("-", "_", option))]] <- args[[index + 1]]
      index <- index + 2
    } else {
      stop(sprintf("Unknown option: %s", option), call. = FALSE)
    }
  }
  result
}

script_path <- function() {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) == 0) {
    return(normalizePath(getwd()))
  }
  dirname(normalizePath(sub("^--file=", "", file_arg[[1]])))
}

rename_columns <- function(data) {
  aliases <- c(
    "jaccard" = "Jaccard",
    "bleu-A" = "BLEU-A",
    "bleu-1" = "BLEU-1",
    "bleu-2" = "BLEU-2",
    "bleu-3" = "BLEU-3",
    "bleu-4" = "BLEU-4",
    "tfidf_cosine" = "TF_IDF_CS",
    "USE_cosine_similarity" = "USE_CS",
    "bert-score-precision" = "BERTScore-P",
    "bert-score-recall" = "BERTScore-R",
    "bert-score-f1" = "BERTScore-F1",
    "sentence_bert_cosine_similarity" = "SentenceBERT_CS",
    "infersent_cosine_similarity" = "InferSent_CS",
    "meteor" = "METEOR",
    "rouge-l-f1" = "ROUGE-L-F1",
    "rouge-1-f1" = "ROUGE-1-F1",
    "rouge-1-p" = "ROUGE-1-P",
    "rouge-1-r" = "ROUGE-1-R",
    "rouge-2-f1" = "ROUGE-2-F1",
    "rouge-2-p" = "ROUGE-2-P",
    "rouge-2-r" = "ROUGE-2-R",
    "rouge-3-f1" = "ROUGE-3-F1",
    "rouge-3-p" = "ROUGE-3-P",
    "rouge-3-r" = "ROUGE-3-R",
    "rouge-4-f1" = "ROUGE-4-F1",
    "rouge-4-p" = "ROUGE-4-P",
    "rouge-4-r" = "ROUGE-4-R",
    "rouge-l-p" = "ROUGE-L-P",
    "rouge-l-r" = "ROUGE-L-R",
    "rouge-w-f1" = "ROUGE-W-F1",
    "rouge-w-p" = "ROUGE-W-P",
    "rouge-w-r" = "ROUGE-W-R",
    "tfidf_euclidean" = "TF_IDF_ED",
    "chrf_score" = "chrF",
    "USE_euclidean_distance" = "USE_ED",
    "sentence_bert_euclidean_distance" = "SentenceBERT_ED",
    "infersent_euclidean_distance" = "InferSent_ED",
    "CodeT5-plus-cosine-similarity" = "CodeT5-plus_CS"
  )

  names(data) <- unname(ifelse(names(data) %in% names(aliases), aliases[names(data)], names(data)))
  data
}

write_latex <- function(table_data, output_path) {
  columns <- names(table_data)
  alignment <- paste0("l", paste(rep("r", length(columns) - 1), collapse = ""))
  escape <- function(value) gsub("_", "\\_", value, fixed = TRUE)
  lines <- c(
    sprintf("\\begin{tabular}{%s}", alignment),
    "\\hline",
    paste(escape(columns), collapse = " & "),
    "\\\\",
    "\\hline"
  )
  for (row in seq_len(nrow(table_data))) {
    values <- vapply(table_data[row, , drop = FALSE], as.character, character(1))
    lines <- c(lines, paste(escape(values), collapse = " & "), "\\\\")
  }
  lines <- c(lines, "\\hline", "\\end{tabular}")
  writeLines(lines, output_path)
}

legacy_table <- function() {
  metric_names <- c(
    "Proportion of Variance", "Cumulative Proportion", "BLEU-1", "BERTScore-R",
    "SentenceBERT_CS", "InferSent_CS", "c_coeff", "ROUGE-1-P", "ROUGE-4-R",
    "ROUGE-W-R", "CodeT5-plus_CS", "SIDE"
  )
  values <- rbind(
    c(0.55354, 0.16130, 0.08394, 0.06976, 0.04125, 0.02905, 0.02490, 0.01840, 0.00986, 0.00799),
    c(0.55354, 0.71484, 0.79879, 0.86855, 0.90980, 0.93886, 0.96375, 0.98215, 0.99201, 1.00000),
    c(0.25691, 0.24039, -0.43554, 0.10524, -0.49778, 0.13209, -0.43546, 0.45272, -0.08973, -0.08706),
    c(0.47315, 0.15115, -0.23913, 0.24509, -0.18134, -0.18081, 0.73901, -0.06954, 0.10237, -0.09488),
    c(0.37621, -0.09982, 0.04275, 0.17239, 0.21846, -0.68241, -0.37313, -0.17663, -0.31678, -0.18105),
    c(0.23058, -0.00883, -0.01141, 0.05606, 0.08748, -0.21245, -0.16869, 0.09707, 0.54195, 0.74759),
    c(0.12728, -0.46613, 0.52989, 0.39306, -0.54442, 0.13188, -0.06539, -0.09636, 0.06122, -0.01150),
    c(0.43965, 0.07067, -0.10066, 0.21306, 0.32804, 0.62706, -0.17172, -0.42833, -0.14347, 0.10881),
    c(0.41420, 0.25647, 0.38294, -0.72842, -0.25301, 0.01104, -0.01560, -0.13604, -0.04738, 0.01699),
    c(0.30421, -0.03528, 0.36132, 0.05836, 0.44045, 0.15132, 0.04829, 0.64465, 0.19471, -0.31551),
    c(-0.00256, -0.02165, -0.16702, -0.07936, -0.02685, -0.01775, -0.23178, -0.33554, 0.72004, -0.52870),
    c(0.20452, -0.78712, -0.39390, -0.39471, 0.03488, 0.07005, 0.08057, 0.10084, -0.06676, 0.02056)
  )
  data.frame(Metric = metric_names, stats::setNames(as.data.frame(values), paste0("PC", 1:10)), check.names = FALSE)
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
`%||%` <- function(left, right) if (is.null(left)) right else left
analysis_dir <- script_path()
repo_root <- normalizePath(file.path(analysis_dir, ".."), mustWork = TRUE)
input_path <- args$input %||% file.path(repo_root, "Results", "run-on-test", "human-annotated-dataset-with-metrics.csv")
output_dir <- args$output_dir %||% file.path(analysis_dir, "output", "table1")

if (!requireNamespace("Hmisc", quietly = TRUE)) {
  stop("Table 1 reproduction requires the R package 'Hmisc' for redun().", call. = FALSE)
}
if (!file.exists(input_path)) {
  stop(sprintf("Input CSV not found: %s", input_path), call. = FALSE)
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

evaluation_metrics <- c(
  "Jaccard", "BLEU-A", "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "TF_IDF_CS", "USE_CS",
  "BERTScore-P", "BERTScore-R", "BERTScore-F1", "SentenceBERT_CS", "InferSent_CS", "METEOR",
  "ROUGE-L-F1", "c_coeff", "ROUGE-1-F1", "ROUGE-1-P", "ROUGE-1-R", "ROUGE-2-F1",
  "ROUGE-2-P", "ROUGE-2-R", "ROUGE-3-F1", "ROUGE-3-P", "ROUGE-3-R", "ROUGE-4-F1",
  "ROUGE-4-P", "ROUGE-4-R", "ROUGE-L-P", "ROUGE-L-R", "ROUGE-W-F1", "ROUGE-W-P",
  "ROUGE-W-R", "TF_IDF_ED", "chrF", "USE_ED", "SentenceBERT_ED", "InferSent_ED",
  "CodeT5-plus_CS", "SIDE"
)

data <- rename_columns(utils::read.csv(input_path, check.names = FALSE))
required <- c("mid", evaluation_metrics)
missing <- setdiff(required, names(data))
if (length(missing) > 0) {
  stop(sprintf("Input is missing required column(s): %s", paste(missing, collapse = ", ")), call. = FALSE)
}

analysis_data <- data[data$mid != 0, evaluation_metrics, drop = FALSE]
analysis_data <- stats::na.omit(analysis_data)
if (nrow(analysis_data) == 0) {
  stop("No complete records remain after applying mid != 0.", call. = FALSE)
}

formula <- stats::as.formula(paste("~", paste(sprintf("`%s`", evaluation_metrics), collapse = " + ")))
reduction <- Hmisc::redun(formula, data = analysis_data, r2 = 0.8, nk = 0)
reduced_metrics <- reduction$In[!is.na(reduction$In)]
if (length(reduced_metrics) == 0) {
  stop("redun() did not retain any metrics.", call. = FALSE)
}

pca <- stats::prcomp(analysis_data[, reduced_metrics, drop = FALSE], scale. = FALSE)
component_count <- ncol(pca$rotation)
component_names <- paste0("PC", seq_len(component_count))
importance <- summary(pca)$importance
reproduced <- rbind(
  importance["Proportion of Variance", , drop = FALSE],
  importance["Cumulative Proportion", , drop = FALSE],
  pca$rotation
)
reproduced <- data.frame(Metric = rownames(reproduced), reproduced, check.names = FALSE, row.names = NULL)
names(reproduced)[-1] <- component_names

legacy <- legacy_table()
utils::write.csv(legacy, file.path(output_dir, "table1-notebook-reference.csv"), row.names = FALSE)
utils::write.csv(reproduced, file.path(output_dir, "table1-reproduced.csv"), row.names = FALSE)
write_latex(reproduced, file.path(output_dir, "table1-reproduced.tex"))
utils::write.csv(data.frame(Metric = reduced_metrics), file.path(output_dir, "table1-reduced-metrics.csv"), row.names = FALSE)

shared_metrics <- intersect(legacy$Metric, reproduced$Metric)
shared_components <- intersect(names(legacy), names(reproduced))
shared_components <- setdiff(shared_components, "Metric")
comparison <- merge(
  legacy[legacy$Metric %in% shared_metrics, c("Metric", shared_components), drop = FALSE],
  reproduced[reproduced$Metric %in% shared_metrics, c("Metric", shared_components), drop = FALSE],
  by = "Metric", suffixes = c("_notebook", "_reproduced")
)
for (component in shared_components) {
  comparison[[paste0(component, "_difference")]] <- comparison[[paste0(component, "_reproduced")]] - comparison[[paste0(component, "_notebook")]]
}
utils::write.csv(comparison, file.path(output_dir, "table1-notebook-vs-reproduced.csv"), row.names = FALSE)

writeLines(c(
  sprintf("Input: %s", normalizePath(input_path)),
  sprintf("Rows after mid != 0 and complete-case filtering: %d", nrow(analysis_data)),
  sprintf("Reduced metrics: %s", paste(reduced_metrics, collapse = ", "))
), file.path(output_dir, "table1-run-summary.txt"))

message(sprintf("[done] Table 1 outputs written to %s", normalizePath(output_dir)))
