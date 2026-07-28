import argparse
import shutil
from pathlib import Path

from common import REPO_ROOT, run_cmd


MODELS_ROOT = REPO_ROOT / "study-2" / "training-sidep" / "models"


def clean_output_dir(output_path: Path) -> None:
    resolved_output = output_path.resolve()
    resolved_models_root = MODELS_ROOT.resolve()

    if resolved_output == resolved_models_root:
        raise ValueError(f"Refusing to clean the models root itself: {resolved_output}")

    try:
        resolved_output.relative_to(resolved_models_root)
    except ValueError as exc:
        raise ValueError(
            "--clean-output only removes directories under "
            f"{resolved_models_root}; got {resolved_output}"
        ) from exc

    if output_path.exists():
        if output_path.is_symlink() or not output_path.is_dir():
            raise ValueError(f"Refusing to clean non-directory output path: {output_path}")
        shutil.rmtree(output_path)
        print(f"[CLEAN] Removed existing SIDE-py output directory: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2: train SIDE-py model.")
    parser.add_argument(
        "--train-file",
        type=Path,
        default=REPO_ROOT / "study-2" / "training-sidep" / "side_finetune_codexglue.json",
    )
    parser.add_argument(
        "--val-file",
        type=Path,
        default=REPO_ROOT / "study-2" / "training-sidep" / "side_finetune_codexglue_valid.json",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=REPO_ROOT / "study-2" / "training-sidep" / "models" / "mpnet_triplet_no_hardneg_v2-test",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--checkpoint-steps", type=int, default=5000)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--python-bin", default="python3")
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete the existing SIDE-py output directory before training.",
    )
    args = parser.parse_args()

    if args.clean_output:
        clean_output_dir(args.output_path)

    script = REPO_ROOT / "study-2" / "training-sidep" / "train_model_for_SIDE-p-new.py"
    run_cmd(
        [
            args.python_bin,
            str(script),
            "--train-file",
            str(args.train_file),
            "--val-file",
            str(args.val_file),
            "--output-path",
            str(args.output_path),
            "--batch-size",
            str(args.batch_size),
            "--epochs",
            str(args.epochs),
            "--max-seq-length",
            str(args.max_seq_length),
            "--patience",
            str(args.patience),
            "--checkpoint-steps",
            str(args.checkpoint_steps),
            "--cuda-visible-devices",
            str(args.cuda_visible_devices),
        ],
        cwd=REPO_ROOT / "study-2" / "training-sidep",
    )
    print("[OK] Training finished.")


if __name__ == "__main__":
    main()
