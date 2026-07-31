import argparse
import csv
import os
import shutil
from datetime import date
from pathlib import Path
from typing import Optional

from common import REPO_ROOT, run_cmd


def threshold_dir_label(threshold: float) -> str:
    return f"{threshold:g}".replace(".", "_")


def threshold_run_label(threshold: float) -> str:
    return f"{threshold:g}".replace(".", "")


def default_run_root(threshold: float, run_date: Optional[str]) -> Path:
    label = threshold_run_label(threshold)
    day = run_date or date.today().isoformat()
    return REPO_ROOT / "study-3" / "replay-runs" / f"{day}-base-hf-side{label}"


def ensure_annotation_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def require_files(paths: list[Path], context: str) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        formatted = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(f"Missing required {context} file(s):\n{formatted}")


def run_train(
    python_bin: str,
    train_file: Path,
    valid_file: Path,
    test_file: Path,
    output_dir: Path,
    cuda_visible_devices: str,
    model_name_or_path: str,
    tokenizer_name: str,
    encoder_block_size: int,
    decoder_block_size: int,
    train_batch_size: int,
    eval_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    adam_epsilon: float,
    max_grad_norm: float,
    epochs: int,
    seed: int,
    num_beams: int,
) -> None:
    run_cmd(
        [
            python_bin,
            str(REPO_ROOT / "pipeline" / "run_train_base_replay.py"),
            "--train-file",
            str(train_file),
            "--valid-file",
            str(valid_file),
            "--test-file",
            str(test_file),
            "--base-output-dir",
            str(output_dir),
            "--cuda-visible-devices",
            str(cuda_visible_devices),
            "--model-name-or-path",
            model_name_or_path,
            "--tokenizer-name",
            tokenizer_name,
            "--encoder-block-size",
            str(encoder_block_size),
            "--decoder-block-size",
            str(decoder_block_size),
            "--train-batch-size",
            str(train_batch_size),
            "--eval-batch-size",
            str(eval_batch_size),
            "--gradient-accumulation-steps",
            str(gradient_accumulation_steps),
            "--learning-rate",
            str(learning_rate),
            "--adam-epsilon",
            str(adam_epsilon),
            "--max-grad-norm",
            str(max_grad_norm),
            "--epochs",
            str(epochs),
            "--seed",
            str(seed),
            "--num-beams",
            str(num_beams),
        ]
    )


def run_infer(
    python_bin: str,
    benchmark_input: Path,
    model_path: Path,
    output_dir: Path,
    prompt_type: str,
    cuda_visible_devices: str,
    tokenizer_name: str,
    encoder_block_size: int,
    decoder_block_size: int,
    eval_batch_size: int,
    num_beams: int,
) -> None:
    run_cmd(
        [
            python_bin,
            str(REPO_ROOT / "pipeline" / "run_infer_base_replay.py"),
            "--input-file",
            str(benchmark_input),
            "--model-path",
            str(model_path),
            "--output-dir",
            str(output_dir),
            "--prompt-types",
            prompt_type,
            "--cuda-visible-devices",
            str(cuda_visible_devices),
            "--tokenizer-name",
            tokenizer_name,
            "--encoder-block-size",
            str(encoder_block_size),
            "--decoder-block-size",
            str(decoder_block_size),
            "--eval-batch-size",
            str(eval_batch_size),
            "--num-beams",
            str(num_beams),
        ]
    )


def run_traditional_metrics(
    python_bin: str,
    input_csv: Path,
    output_csv: Path,
    summary_txt: Path,
) -> None:
    run_cmd(
        [
            python_bin,
            str(REPO_ROOT / "pipeline" / "run_compare_metrics.py"),
            "--input-csv",
            str(input_csv),
            "--output-csv",
            str(output_csv),
            "--summary-txt",
            str(summary_txt),
        ]
    )


def run_annotation_side(
    python_bin: str,
    annotation_csv: Path,
    output_csv: Path,
    side_checkpoint: Path,
) -> None:
    run_cmd(
        [
            python_bin,
            str(REPO_ROOT / "pipeline" / "run_score_sidepy.py"),
            "--csv-input",
            str(annotation_csv),
            "--csv-output",
            str(output_csv),
            "--checkpoint",
            str(side_checkpoint),
            "--code-col",
            "codeFunctions",
            "--summary-col",
            "codeComment",
            "--score-col",
            "SIDE_score",
        ]
    )


def run_annotation_regression_metrics(
    python_bin: str,
    annotation_csv: Path,
    output_csv: Path,
) -> None:
    """Refresh Table 2 predictors after replacing codeComment."""
    run_cmd(
        [
            python_bin,
            str(REPO_ROOT / "study-2" / "training-sidep" / "computAllMetrics.py"),
            "--input-csv",
            str(annotation_csv),
            "--output-csv",
            str(output_csv),
        ],
        cwd=REPO_ROOT / "study-2" / "training-sidep",
    )


def attach_side_scores(inference_csv: Path, side_csv: Path, output_csv: Path) -> None:
    with inference_csv.open("r", encoding="utf-8", errors="ignore", newline="") as inference_handle:
        inference_reader = csv.DictReader(inference_handle)
        inference_fields = list(inference_reader.fieldnames or [])
        inference_rows = list(inference_reader)

    with side_csv.open("r", encoding="utf-8", errors="ignore", newline="") as side_handle:
        side_reader = csv.DictReader(side_handle)
        side_fields = list(side_reader.fieldnames or [])
        side_rows = list(side_reader)

    if "SIDE_score" not in side_fields:
        raise ValueError(f"Missing SIDE_score in {side_csv}")
    if len(inference_rows) != len(side_rows):
        raise ValueError(
            f"Row count mismatch while attaching SIDE scores: "
            f"inference={len(inference_rows)} side={len(side_rows)}"
        )

    output_fields = list(inference_fields)
    if "SIDE_score" not in output_fields:
        output_fields.append("SIDE_score")

    for inference_row, side_row in zip(inference_rows, side_rows):
        inference_row["SIDE_score"] = side_row["SIDE_score"]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(inference_rows)

    print(f"[OK] Attached annotation SIDE scores to inference metrics CSV: {output_csv}")


def run_llm_judge(
    python_bin: str,
    input_csv: Path,
    output_csv: Path,
    summary_txt: Path,
    model_name: str,
    cuda_visible_devices: str,
    resume: bool,
) -> None:
    run_cmd(
        [
            "env",
            "PYTHONNOUSERSITE=1",
            python_bin,
            str(REPO_ROOT / "pipeline" / "run_llm_judge_metric.py"),
            "--input-csv",
            str(input_csv),
            "--output-csv",
            str(output_csv),
            "--summary-txt",
            str(summary_txt),
            "--model-name",
            model_name,
            "--cuda-visible-devices",
            cuda_visible_devices,
            *(["--resume"] if resume else []),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train the two CodeT5+ 770m Base replay models and evaluate them under a dated replay directory."
        )
    )
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument(
        "--run-date",
        default=None,
        help="Date label for the auto-created run root. Defaults to today's date.",
    )
    parser.add_argument(
        "--eval-mode",
        choices=["traditional", "llm", "llm-only", "none"],
        default="traditional",
        help=(
            "traditional: fresh annotation SIDE plus BLEU/ROUGE/METEOR/ChrF/TF-IDF with Spearman. "
            "llm: traditional evaluation plus LLM judge. "
            "llm-only: run only LLM judge from existing fresh SIDE annotation CSVs. "
            "none: train and infer only."
        ),
    )
    parser.add_argument(
        "--eval-target",
        choices=["both", "no-side", "with-side"],
        default="both",
        help="Which arm(s) to run inference/evaluation for.",
    )
    parser.add_argument("--repo-id", default="apcl/funcom-python")
    parser.add_argument(
        "--side-checkpoint",
        type=Path,
        default=REPO_ROOT / "study-2" / "training-sidep" / "models" / "mpnet_triplet_no_hardneg_v2-test",
    )
    parser.add_argument(
        "--benchmark-input",
        type=Path,
        default=None,
        help=(
            "Optional JSONL benchmark input for inference. If omitted, this pipeline builds one from "
            "--source-annotation-csv using codeFunctions/originalComment."
        ),
    )
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--python-bin", default="python")
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--model-name-or-path", default="Salesforce/codet5p-770m")
    parser.add_argument("--tokenizer-name", default="Salesforce/codet5p-770m")
    parser.add_argument("--encoder-block-size", type=int, default=512)
    parser.add_argument("--decoder-block-size", type=int, default=128)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--judge-model-name", default="openai/gpt-oss-20b")
    parser.add_argument("--judge-cuda-visible-devices", default="2,3")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument(
        "--judge-python-bin",
        default=None,
        help="Optional Python executable for LLM judge. Defaults to --python-bin.",
    )
    parser.add_argument("--data-no-side-raw-dir", type=Path, default=None)
    parser.add_argument("--data-no-side-dir", type=Path, default=None)
    parser.add_argument("--data-with-side-dir", type=Path, default=None)
    parser.add_argument(
        "--source-annotation-csv",
        type=Path,
        default=REPO_ROOT / "study-2" / "data-files" / "extension" / "gpt-human_annotation-500.csv",
    )
    parser.add_argument("--annotation-no-side-csv", type=Path, default=None)
    parser.add_argument("--annotation-with-side-csv", type=Path, default=None)
    parser.add_argument(
        "--skip-data-stages",
        action="store_true",
        help="Data is already prepared; skip export, SIDE filtering, and no-side matching.",
    )
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-filter", action="store_true")
    parser.add_argument("--skip-match-no-side", action="store_true")
    parser.add_argument("--skip-train-no-side", action="store_true")
    parser.add_argument("--skip-train-with-side", action="store_true")
    parser.add_argument("--skip-infer-no-side", action="store_true")
    parser.add_argument("--skip-infer-with-side", action="store_true")
    parser.add_argument("--skip-replace-codecomment", action="store_true")
    parser.add_argument(
        "--skip-regression-metrics",
        action="store_true",
        help="Use existing recomputed annotation metrics instead of regenerating Table 2 predictors.",
    )
    parser.add_argument("--skip-score-annotation-side", action="store_true")
    parser.add_argument("--skip-llm-judge", action="store_true")
    parser.add_argument("--no-llm-judge-resume", action="store_true")
    parser.add_argument("--skip-compare-metrics", action="store_true")
    parser.add_argument(
        "--skip-score-inference",
        action="store_true",
        help="Deprecated no-op. Inference CSVs do not contain code, so SIDE is scored on annotation CSVs only.",
    )
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    threshold_label = threshold_dir_label(args.threshold)
    run_root = args.run_root or default_run_root(args.threshold, args.run_date)
    args.run_root = run_root

    data_no_side_raw = args.data_no_side_raw_dir or (run_root / "data" / "hf-no-side")
    data_no_side = args.data_no_side_dir or (run_root / "data" / f"hf-no-side-matched-threshold-{threshold_label}")
    data_with_side = args.data_with_side_dir or (run_root / "data" / f"hf-with-side-threshold-{threshold_label}")
    model_no_side = run_root / "models" / "no-side"
    model_with_side = run_root / "models" / f"with-side-threshold-{threshold_label}"
    infer_no_side = run_root / "inference" / "no-side"
    infer_with_side = run_root / "inference" / f"with-side-threshold-{threshold_label}"
    annotation_no_side = args.annotation_no_side_csv or (
        run_root / "evaluation" / "500-human-annotation" / "no-side_500-human-annotation.csv"
    )
    annotation_with_side = args.annotation_with_side_csv or (
        run_root
        / "evaluation"
        / "500-human-annotation"
        / f"with-side-threshold-{threshold_label}_500-human-annotation.csv"
    )
    benchmark_input = args.benchmark_input or (
        run_root / "evaluation" / "500-human-annotation" / "gpt-human_annotation-500.inference.jsonl"
    )
    metrics_no_side = run_root / "evaluation" / "metrics" / "no-side"
    metrics_with_side = run_root / "evaluation" / "metrics" / f"with-side-threshold-{threshold_label}"
    judge_python_bin = args.judge_python_bin or args.python_bin

    prompt_no_side = "base_hf_no_side"
    prompt_with_side = f"base_hf_with_side_{threshold_label}"
    eval_no_side = args.eval_target in {"both", "no-side"}
    eval_with_side = args.eval_target in {"both", "with-side"}
    infer_csv_no_side = infer_no_side / f"{prompt_no_side}_benchmark_inference_results-test.csv"
    infer_csv_with_side = infer_with_side / f"{prompt_with_side}_benchmark_inference_results-test.csv"
    scored_no_side = metrics_no_side / "base_hf_no_side_with_side_scores.csv"
    scored_with_side = metrics_with_side / f"base_hf_with_side_{threshold_label}_with_side_scores.csv"
    metrics_csv_no_side = metrics_no_side / "base_hf_no_side_with_metrics.csv"
    metrics_csv_with_side = metrics_with_side / f"base_hf_with_side_{threshold_label}_with_metrics.csv"
    metrics_summary_no_side = metrics_no_side / "base_hf_no_side_metrics.summary.txt"
    metrics_summary_with_side = metrics_with_side / f"base_hf_with_side_{threshold_label}_metrics.summary.txt"

    annotation_scored_no_side = metrics_no_side / "no-side_500-human-annotation_with_fresh_side.csv"
    annotation_scored_with_side = (
        metrics_with_side / f"with-side-threshold-{threshold_label}_500-human-annotation_with_fresh_side.csv"
    )
    annotation_metrics_no_side = metrics_no_side / "no-side_500-human-annotation_with_regression_metrics.csv"
    annotation_metrics_with_side = (
        metrics_with_side / f"with-side-threshold-{threshold_label}_500-human-annotation_with_regression_metrics.csv"
    )
    llm_judge_no_side = metrics_no_side / "no-side_500-human-annotation_with_fresh_side_llm_judge.csv"
    llm_judge_with_side = (
        metrics_with_side / f"with-side-threshold-{threshold_label}_500-human-annotation_with_fresh_side_llm_judge.csv"
    )

    skip_export = args.skip_export or args.skip_data_stages
    skip_filter = args.skip_filter or args.skip_data_stages
    skip_match_no_side = args.skip_match_no_side or args.skip_data_stages

    print(f"[PLAN] run_root={run_root}")
    print(f"[PLAN] eval_mode={args.eval_mode}")
    print(f"[PLAN] eval_target={args.eval_target}")
    print(f"[PLAN] model={args.model_name_or_path}")
    print(f"[PLAN] CUDA_VISIBLE_DEVICES={args.cuda_visible_devices} (inside subprocesses this appears as cuda:0)")
    print(
        "[PLAN] train="
        f"{'no-side ' if not args.skip_train_no_side else ''}"
        f"{'with-side' if not args.skip_train_with_side else ''}"
    )
    print(f"[PLAN] data stages={'skipped' if args.skip_data_stages else 'enabled'}")
    print(
        "[PLAN] train params: "
        f"epochs={args.epochs}, train_batch_size={args.train_batch_size}, "
        f"eval_batch_size={args.eval_batch_size}, "
        f"gradient_accumulation_steps={args.gradient_accumulation_steps}, "
        f"effective_train_batch_size={args.train_batch_size * args.gradient_accumulation_steps}, "
        f"lr={args.learning_rate}, "
        f"seed={args.seed}, num_beams={args.num_beams}"
    )

    if args.eval_mode != "llm-only" and not skip_export:
        run_cmd(
            [
                args.python_bin,
                str(REPO_ROOT / "pipeline" / "run_base_data_prep_hf.py"),
                "--repo-id",
                args.repo_id,
                "--output-dir",
                str(data_no_side_raw),
                *([] if args.max_records is None else ["--max-records", str(args.max_records)]),
            ]
        )

    if args.eval_mode != "llm-only" and not skip_filter:
        run_cmd(
            [
                args.python_bin,
                str(REPO_ROOT / "pipeline" / "run_base_side_filter.py"),
                "--input-dir",
                str(data_no_side_raw),
                "--output-dir",
                str(data_with_side),
                "--checkpoint",
                str(args.side_checkpoint),
                "--threshold",
                str(args.threshold),
            ]
        )

    if args.eval_mode != "llm-only" and not skip_match_no_side:
        run_cmd(
            [
                args.python_bin,
                str(REPO_ROOT / "pipeline" / "run_base_match_filtered_size.py"),
                "--source-dir",
                str(data_no_side_raw),
                "--target-dir",
                str(data_with_side),
                "--output-dir",
                str(data_no_side),
                "--seed",
                str(args.sample_seed),
            ]
        )

    if args.eval_mode != "llm-only":
        required_training_files = []
        if not args.skip_train_no_side:
            required_training_files.extend(
                [
                    data_no_side / "train.jsonl",
                    data_no_side / "valid.jsonl",
                    data_no_side / "test.jsonl",
                ]
            )
        if not args.skip_train_with_side:
            required_training_files.extend(
                [
                    data_with_side / "train.jsonl",
                    data_with_side / "valid.jsonl",
                    data_with_side / "test.jsonl",
                ]
            )
        require_files(required_training_files, "prepared training data")

    if args.eval_mode != "llm-only" and not args.skip_train_no_side:
        run_train(
            args.python_bin,
            data_no_side / "train.jsonl",
            data_no_side / "valid.jsonl",
            data_no_side / "test.jsonl",
            model_no_side,
            args.cuda_visible_devices,
            args.model_name_or_path,
            args.tokenizer_name,
            args.encoder_block_size,
            args.decoder_block_size,
            args.train_batch_size,
            args.eval_batch_size,
            args.gradient_accumulation_steps,
            args.learning_rate,
            args.adam_epsilon,
            args.max_grad_norm,
            args.epochs,
            args.seed,
            args.num_beams,
        )

    if args.eval_mode != "llm-only" and not args.skip_train_with_side:
        run_train(
            args.python_bin,
            data_with_side / "train.jsonl",
            data_with_side / "valid.jsonl",
            data_with_side / "test.jsonl",
            model_with_side,
            args.cuda_visible_devices,
            args.model_name_or_path,
            args.tokenizer_name,
            args.encoder_block_size,
            args.decoder_block_size,
            args.train_batch_size,
            args.eval_batch_size,
            args.gradient_accumulation_steps,
            args.learning_rate,
            args.adam_epsilon,
            args.max_grad_norm,
            args.epochs,
            args.seed,
            args.num_beams,
        )

    model_name = args.model_name_or_path.split("/")[-1]
    model_dir_name_no_side = f"{model_name}_{(data_no_side / 'train.jsonl').stem}"
    model_dir_name_with_side = f"{model_name}_{(data_with_side / 'train.jsonl').stem}"
    will_infer_no_side = args.eval_mode != "llm-only" and eval_no_side and not args.skip_infer_no_side
    will_infer_with_side = args.eval_mode != "llm-only" and eval_with_side and not args.skip_infer_with_side

    if (
        args.eval_mode != "llm-only"
        and args.benchmark_input is None
        and (will_infer_no_side or will_infer_with_side)
    ):
        run_cmd(
            [
                args.python_bin,
                str(REPO_ROOT / "pipeline" / "run_annotation_csv_to_inference_jsonl.py"),
                "--input-csv",
                str(args.source_annotation_csv),
                "--output-jsonl",
                str(benchmark_input),
            ]
        )

    if will_infer_no_side:
        run_infer(
            args.python_bin,
            benchmark_input,
            model_no_side / model_dir_name_no_side / "final_model",
            infer_no_side,
            prompt_no_side,
            args.cuda_visible_devices,
            args.tokenizer_name,
            args.encoder_block_size,
            args.decoder_block_size,
            args.eval_batch_size,
            args.num_beams,
        )

    if will_infer_with_side:
        run_infer(
            args.python_bin,
            benchmark_input,
            model_with_side / model_dir_name_with_side / "final_model",
            infer_with_side,
            prompt_with_side,
            args.cuda_visible_devices,
            args.tokenizer_name,
            args.encoder_block_size,
            args.decoder_block_size,
            args.eval_batch_size,
            args.num_beams,
        )

    if args.eval_mode in {"traditional", "llm"} and (
        not args.skip_replace_codecomment or not args.skip_compare_metrics
    ):
        required_inference_csvs = []
        if eval_no_side:
            required_inference_csvs.append(infer_csv_no_side)
        if eval_with_side:
            required_inference_csvs.append(infer_csv_with_side)
        require_files(
            required_inference_csvs,
            (
                "inference output for evaluation. Rerun without the matching "
                "--skip-infer-* flag, or also skip replacement and metric comparison"
            ),
        )

    if args.eval_mode in {"traditional", "llm"} and not args.skip_replace_codecomment:
        if eval_no_side:
            ensure_annotation_copy(args.source_annotation_csv, annotation_no_side)
            run_cmd(
                [
                    args.python_bin,
                    str(REPO_ROOT / "pipeline" / "run_replace_codecomment_from_inference.py"),
                    "--annotation-csv",
                    str(annotation_no_side),
                    "--inference-csv",
                    str(infer_csv_no_side),
                ]
            )
        if eval_with_side:
            ensure_annotation_copy(args.source_annotation_csv, annotation_with_side)
            run_cmd(
                [
                    args.python_bin,
                    str(REPO_ROOT / "pipeline" / "run_replace_codecomment_from_inference.py"),
                    "--annotation-csv",
                    str(annotation_with_side),
                    "--inference-csv",
                    str(infer_csv_with_side),
                ]
            )

    if args.eval_mode in {"traditional", "llm"}:
        required_annotations = []
        if eval_no_side:
            required_annotations.append(annotation_no_side)
        if eval_with_side:
            required_annotations.append(annotation_with_side)
        require_files(required_annotations, "annotation CSV with condition-specific predictions")
        if not args.skip_regression_metrics:
            if eval_no_side:
                metrics_no_side.mkdir(parents=True, exist_ok=True)
                run_annotation_regression_metrics(args.python_bin, annotation_no_side, annotation_metrics_no_side)
            if eval_with_side:
                metrics_with_side.mkdir(parents=True, exist_ok=True)
                run_annotation_regression_metrics(args.python_bin, annotation_with_side, annotation_metrics_with_side)

    if args.eval_mode in {"traditional", "llm"} and not args.skip_score_annotation_side:
        required_regression_metric_csvs = []
        if eval_no_side:
            required_regression_metric_csvs.append(annotation_metrics_no_side)
        if eval_with_side:
            required_regression_metric_csvs.append(annotation_metrics_with_side)
        require_files(required_regression_metric_csvs, "recomputed annotation metric CSV")
        if eval_no_side:
            metrics_no_side.mkdir(parents=True, exist_ok=True)
            run_annotation_side(args.python_bin, annotation_metrics_no_side, annotation_scored_no_side, args.side_checkpoint)
        if eval_with_side:
            metrics_with_side.mkdir(parents=True, exist_ok=True)
            run_annotation_side(args.python_bin, annotation_metrics_with_side, annotation_scored_with_side, args.side_checkpoint)

    if args.eval_mode in {"traditional", "llm"} and not args.skip_compare_metrics:
        required_side_csvs = []
        if eval_no_side:
            required_side_csvs.append(annotation_scored_no_side)
        if eval_with_side:
            required_side_csvs.append(annotation_scored_with_side)
        require_files(
            required_side_csvs,
            (
                "fresh annotation SIDE score output for metric comparison. Rerun without "
                "--skip-score-annotation-side, or also skip metric comparison"
            ),
        )
        if eval_no_side:
            metrics_no_side.mkdir(parents=True, exist_ok=True)
            attach_side_scores(infer_csv_no_side, annotation_scored_no_side, scored_no_side)
            run_traditional_metrics(args.python_bin, scored_no_side, metrics_csv_no_side, metrics_summary_no_side)
        if eval_with_side:
            metrics_with_side.mkdir(parents=True, exist_ok=True)
            attach_side_scores(infer_csv_with_side, annotation_scored_with_side, scored_with_side)
            run_traditional_metrics(args.python_bin, scored_with_side, metrics_csv_with_side, metrics_summary_with_side)

    if args.eval_mode in {"llm", "llm-only"} and not args.skip_llm_judge:
        if eval_no_side:
            run_llm_judge(
                judge_python_bin,
                annotation_scored_no_side,
                llm_judge_no_side,
                metrics_no_side / "no-side_500-human-annotation_llm_judge.summary.txt",
                args.judge_model_name,
                args.judge_cuda_visible_devices,
                resume=not args.no_llm_judge_resume,
            )
        if eval_with_side:
            run_llm_judge(
                judge_python_bin,
                annotation_scored_with_side,
                llm_judge_with_side,
                metrics_with_side / f"with-side-threshold-{threshold_label}_500-human-annotation_llm_judge.summary.txt",
                args.judge_model_name,
                args.judge_cuda_visible_devices,
                resume=not args.no_llm_judge_resume,
            )

    print(f"[OK] Base replay outputs live under: {run_root}")


if __name__ == "__main__":
    main()
