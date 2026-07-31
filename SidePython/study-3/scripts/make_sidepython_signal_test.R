#!/usr/bin/env Rscript

# Creates a synthetic-only test copy. It must never be used for a manuscript
# result. The output preserves exactly the input CSV schema: only the three
# analysis-label values are changed, and only in the separately named output.

parse_args <- function(args) {
  result <- list(input = NULL, output = NULL, strength = 0.85)
  index <- 1
  while (index <= length(args)) {
    key <- args[[index]]
    if (!(key %in% c("--input", "--output", "--strength"))) stop(sprintf("Unknown option: %s", key), call. = FALSE)
    if (index == length(args)) stop(sprintf("Missing value for %s", key), call. = FALSE)
    result[[sub("^--", "", key)]] <- args[[index + 1]]
    index <- index + 2
  }
  result$strength <- as.numeric(result$strength)
  if (!is.finite(result$strength) || result$strength < 0 || result$strength > 1) stop("--strength must be in [0, 1].", call. = FALSE)
  result
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
if (is.null(args$input) || is.null(args$output)) stop("Usage: Rscript make_sidepython_signal_test.R --input INPUT.csv --output OUTPUT.csv [--strength 0.85]", call. = FALSE)
if (!file.exists(args$input)) stop(sprintf("Input CSV not found: %s", args$input), call. = FALSE)

data <- utils::read.csv(args$input, check.names = FALSE)
side_column <- if ("SIDE_score" %in% names(data)) "SIDE_score" else if ("SIDEpython" %in% names(data)) "SIDEpython" else NA_character_
targets <- c("human_content_adequacy", "human_conciseness", "human_fluency")
if (is.na(side_column)) stop("Input needs SIDE_score or SIDEpython.", call. = FALSE)
missing <- setdiff(targets, names(data))
if (length(missing)) stop(sprintf("Input is missing: %s", paste(missing, collapse = ", ")), call. = FALSE)

side <- suppressWarnings(as.numeric(data[[side_column]]))
if (any(!is.finite(side))) stop("SIDE scores must all be finite for this synthetic test.", call. = FALSE)
# Average ranks create a reproducible [1, 5] monotone signal without random
# sampling.  The test response is a convex blend of the original rating and
# that signal, then rounded to the ordinal scale used by Table 2.
side_signal <- 1 + 4 * (rank(side, ties.method = "average") - 1) / (length(side) - 1)
for (target in targets) {
  original <- suppressWarnings(as.numeric(data[[target]]))
  if (any(!is.finite(original))) stop(sprintf("%s contains non-finite values.", target), call. = FALSE)
  data[[target]] <- round(pmin(5, pmax(1, (1 - args$strength) * original + args$strength * side_signal)))
}

utils::write.csv(data, args$output, row.names = FALSE)
message(sprintf("[synthetic test only; schema unchanged] wrote %d rows to %s", nrow(data), normalizePath(args$output, mustWork = FALSE)))
