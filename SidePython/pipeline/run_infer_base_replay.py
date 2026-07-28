import argparse
import json
import os
from pathlib import Path

from common import REPO_ROOT, run_cmd


def validate_input_file(input_file: Path, input_field: str, target_field: str) -> None:
    if not input_file.exists():
        raise FileNotFoundError(f"Input benchmark file does not exist: {input_file}")

    with input_file.open("r", encoding="utf-8") as handle:
        first_line = handle.readline().strip()

    if not first_line:
        raise ValueError(f"Input benchmark file is empty: {input_file}")

    if first_line == "version https://git-lfs.github.com/spec/v1":
        raise ValueError(
            "Input benchmark file is a Git LFS pointer, not the real JSONL data: "
            f"{input_file}\n"
            "Run from the repo root:\n"
            '  git lfs pull --include="study-3/evaluation-benchmark-data/shortening_benchmark.jsonl"'
        )

    try:
        sample = json.loads(first_line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Input benchmark file is not valid JSONL at first line: {input_file}") from exc

    missing_fields = [field for field in (input_field, target_field) if field not in sample]
    if missing_fields:
        raise ValueError(
            f"Input benchmark file is missing required field(s) {missing_fields}: {input_file}. "
            f"Available first-row fields: {sorted(sample.keys())}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Base benchmark inference into a fresh replay directory.")
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-types", default="base")
    parser.add_argument("--input-field", default="code_tokens")
    parser.add_argument("--target-field", default="docstring_tokens")
    parser.add_argument("--tokenizer-name", default="Salesforce/codet5p-770m")
    parser.add_argument("--encoder-block-size", type=int, default=512)
    parser.add_argument("--decoder-block-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--python-bin", default="python")
    args = parser.parse_args()

    script = REPO_ROOT / "study-3" / "scripts" / "run_inference.py"
    validate_input_file(args.input_file, args.input_field, args.target_field)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    run_cmd(
        [
            args.python_bin,
            str(script),
            "--input_file",
            str(args.input_file),
            "--model_path",
            str(args.model_path),
            "--output_dir",
            str(args.output_dir),
            "--prompt_types",
            args.prompt_types,
            "--input-field",
            args.input_field,
            "--target-field",
            args.target_field,
            "--tokenizer_name",
            args.tokenizer_name,
            "--encoder_block_size",
            str(args.encoder_block_size),
            "--decoder_block_size",
            str(args.decoder_block_size),
            "--eval_batch_size",
            str(args.eval_batch_size),
            "--num_beams",
            str(args.num_beams),
        ],
        cwd=REPO_ROOT / "study-3" / "scripts",
    )


if __name__ == "__main__":
    main()
