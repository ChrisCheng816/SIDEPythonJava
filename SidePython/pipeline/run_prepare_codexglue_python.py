import argparse
import json
from pathlib import Path

from datasets import load_dataset

from common import REPO_ROOT


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "study-2"
    / "data-files"
    / "codexglue-code-to-text"
    / "dataset"
    / "python"
    / "train.jsonl"
)


def write_jsonl(dataset, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare CodeXGLUE code-to-text Python train.jsonl from Hugging Face."
    )
    parser.add_argument("--dataset-name", default="google/code_x_glue_ct_code_to_text")
    parser.add_argument("--config-name", default="python")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate the output file even if it already exists.",
    )
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        print(f"[OK] CodeXGLUE Python data already exists: {args.output}")
        print(f"[OK] Examples: {count_lines(args.output)}")
        return

    print(f"[LOAD] {args.dataset_name} / {args.config_name} / {args.split}")
    dataset = load_dataset(args.dataset_name, args.config_name, split=args.split)
    print(f"[WRITE] {args.output}")
    write_jsonl(dataset, args.output)

    print(f"[OK] CodeXGLUE Python data: {args.output}")
    print(f"[OK] Examples: {count_lines(args.output)}")


if __name__ == "__main__":
    main()
