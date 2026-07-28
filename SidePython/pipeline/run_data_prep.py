import argparse
from pathlib import Path

from common import REPO_ROOT, run_cmd


DEFAULT_CODEXGLUE_PYTHON_TRAIN = (
    REPO_ROOT
    / "study-2"
    / "data-files"
    / "codexglue-code-to-text"
    / "dataset"
    / "python"
    / "train.jsonl"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: prepare SIDE-py triplet data.")
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=DEFAULT_CODEXGLUE_PYTHON_TRAIN,
        help=(
            "Input CodeXGLUE-style dataset path. Defaults to the preprocessed "
            "CodeXGLUE Python train.jsonl generated from Hugging Face by "
            "run_prepare_codexglue_python.py."
        ),
    )
    parser.add_argument(
        "--all-output",
        type=Path,
        default=REPO_ROOT / "study-2" / "training-sidep" / "side_finetune_codexglue_train-new.json",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=REPO_ROOT / "study-2" / "training-sidep" / "side_finetune_codexglue.json",
    )
    parser.add_argument(
        "--valid-output",
        type=Path,
        default=REPO_ROOT / "study-2" / "training-sidep" / "side_finetune_codexglue_valid.json",
    )
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--sample-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--python-bin", default="python3")
    args = parser.parse_args()

    if not args.input_jsonl.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {args.input_jsonl}. "
            "Run `conda run -n side-py python pipeline/run_prepare_codexglue_python.py` first."
        )

    script = REPO_ROOT / "study-2" / "training-sidep" / "prepare_finetune_data.py"

    run_cmd(
        [
            args.python_bin,
            str(script),
            "--input",
            str(args.input_jsonl),
            "--output",
            str(args.all_output),
            "--train-output",
            str(args.train_output),
            "--valid-output",
            str(args.valid_output),
            "--valid-ratio",
            str(args.valid_ratio),
            "--sample-ratio",
            str(args.sample_ratio),
            "--seed",
            str(args.seed),
        ],
        cwd=REPO_ROOT / "study-2" / "training-sidep",
    )

    for path in [args.all_output, args.train_output, args.valid_output]:
        if not path.exists():
            raise FileNotFoundError(f"Expected output not found: {path}")
    print("[OK] Data preparation finished.")


if __name__ == "__main__":
    main()
