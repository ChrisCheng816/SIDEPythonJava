import argparse
from pathlib import Path

from common import REPO_ROOT, run_cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3: score predictions with SIDE-py.")
    parser.add_argument("--csv-input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--code-col", default="target")
    parser.add_argument("--summary-col", default="raw_predictions")
    parser.add_argument("--score-col", default="SIDE_score")
    parser.add_argument("--run-text-eval", action="store_true")
    parser.add_argument(
        "--metrics-output-dir",
        type=Path,
        default=REPO_ROOT / "study-3" / "inference-results" / "metrics",
    )
    parser.add_argument("--python-bin", default="python3")
    args = parser.parse_args()

    use_side_script = REPO_ROOT / "study-2" / "training-sidep" / "use_SIDE.py"
    csv_output = args.csv_output if args.csv_output else args.csv_input

    run_cmd(
        [
            args.python_bin,
            str(use_side_script),
            "--csv-input",
            str(args.csv_input),
            "--csv-output",
            str(csv_output),
            "--checkpoint",
            str(args.checkpoint),
            "--code-col",
            args.code_col,
            "--summary-col",
            args.summary_col,
            "--score-col",
            args.score_col,
        ],
        cwd=REPO_ROOT / "study-2" / "training-sidep",
    )

    if args.run_text_eval:
        eval_script = REPO_ROOT / "study-3" / "scripts" / "main_stat_codereval.py"
        args.metrics_output_dir.mkdir(parents=True, exist_ok=True)
        run_cmd(
            [
                args.python_bin,
                str(eval_script),
                "--prediction-path",
                str(csv_output),
                "--output-file",
                str(args.metrics_output_dir),
            ],
            cwd=REPO_ROOT / "study-3" / "scripts",
        )

    print("[OK] SIDE scoring finished.")


if __name__ == "__main__":
    main()

