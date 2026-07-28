import argparse
import csv
import json
import random
from pathlib import Path
from typing import List


def extract_inline_comments(code: str) -> List[str]:
    """Extract inline comments that start with '#' but are not full-line comments."""
    return [
        line.split("#", 1)[1].strip()
        for line in code.split("\n")
        if "#" in line and not line.strip().startswith("#")
    ]


def count_code_lines(code: str) -> int:
    """Count non-empty, non-comment lines of code."""
    return len(
        [
            line
            for line in code.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    )


def _looks_like_lfs_pointer(path: Path) -> bool:
    with path.open("r", encoding="utf-8-sig") as f:
        first_line = f.readline().strip()
    return first_line == "version https://git-lfs.github.com/spec/v1"


def _tokens_to_text(value) -> str:
    if isinstance(value, list):
        return " ".join(str(x) for x in value)

    value = (value or "").strip()
    if not value:
        return ""

    # Accept JSON list cells if present, otherwise keep plain tokenized text.
    if value.startswith("[") and value.endswith("]"):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return " ".join(str(x) for x in parsed)
        except json.JSONDecodeError:
            pass
    return value


def load_dataset(path: Path) -> list:
    if _looks_like_lfs_pointer(path):
        raise ValueError(
            f"Input is a Git LFS pointer, not real data: {path}. "
            "Please provide a real JSONL/CSV file."
        )

    suffix = path.suffix.lower()
    if suffix == ".csv":
        rows = []
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            required = {"code_tokens", "docstring_tokens"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"CSV missing columns: {sorted(missing)}")

            for row in reader:
                rows.append(
                    {
                        "code_tokens": _tokens_to_text(row.get("code_tokens", "")),
                        "docstring_tokens": _tokens_to_text(row.get("docstring_tokens", "")),
                    }
                )
        return rows

    with path.open("r", encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f]


def build_side_triplets(dataset: list, seed: int = 42) -> list:
    random.seed(seed)
    side_data = []
    banned_markers = [
        "todo",
        "to-do",
        "fixme",
        "fix-me",
        "xxx",
        "hackme",
        "hack-me",
        "debug",
        "remove",
    ]

    dataset_size = len(dataset)
    for i, entry in enumerate(dataset):
        code = _tokens_to_text(entry["code_tokens"])
        pos = _tokens_to_text(entry["docstring_tokens"])

        neg_idx = random.randrange(dataset_size - 1)
        if neg_idx >= i:
            neg_idx += 1
        neg = _tokens_to_text(dataset[neg_idx]["docstring_tokens"])

        total_code_lines = count_code_lines(code)
        inline_comments = extract_inline_comments(code)
        hard_negatives = []
        for comment in inline_comments:
            if any(marker in comment.lower() for marker in banned_markers):
                continue
            if total_code_lines == 0:
                continue
            if (1 / total_code_lines) < 0.25:
                hard_negatives.append(comment)

        side_data.append(
            {
                "query": code,
                "pos": pos.strip(),
                "neg": neg.strip(),
                "hardNegative": hard_negatives,
            }
        )
    return side_data


def sample_dataset(dataset: list, sample_ratio: float, seed: int) -> list:
    if sample_ratio >= 1.0:
        return dataset

    sample_size = int(len(dataset) * sample_ratio)
    if sample_size < 2:
        raise ValueError(
            f"--sample-ratio={sample_ratio} leaves fewer than 2 examples from {len(dataset)} rows."
        )

    rng = random.Random(seed)
    sampled_indices = sorted(rng.sample(range(len(dataset)), sample_size))
    return [dataset[i] for i in sampled_indices]


def dump_json(data: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Build SIDE triplet training data from a JSONL code-summary dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("../data-files/codexglue-code-to-text/dataset/python/train.jsonl"),
        help="Input JSONL/CSV file with code_tokens and docstring_tokens.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("side_finetune_codexglue_train-new.json"),
        help="Output JSON file containing all generated triplets.",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=Path("side_finetune_codexglue.json"),
        help="Train split output JSON path.",
    )
    parser.add_argument(
        "--valid-output",
        type=Path,
        default=Path("side_finetune_codexglue_valid.json"),
        help="Validation split output JSON path.",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.1,
        help="Validation ratio for train/valid split.",
    )
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=0.5,
        help="Deterministic random fraction of the source dataset to keep before splitting.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    dataset = load_dataset(args.input)
    if len(dataset) < 2:
        raise ValueError("Input dataset must contain at least 2 rows.")
    if not 0.0 < args.valid_ratio < 1.0:
        raise ValueError("--valid-ratio must be in (0, 1).")
    if not 0.0 < args.sample_ratio <= 1.0:
        raise ValueError("--sample-ratio must be in (0, 1].")

    original_size = len(dataset)
    dataset = sample_dataset(dataset, args.sample_ratio, args.seed)
    side_data = build_side_triplets(dataset, seed=args.seed)
    dump_json(side_data, args.output)

    random.seed(args.seed)
    random.shuffle(side_data)
    split_idx = int(len(side_data) * (1 - args.valid_ratio))
    train_data = side_data[:split_idx]
    valid_data = side_data[split_idx:]
    dump_json(train_data, args.train_output)
    dump_json(valid_data, args.valid_output)

    hard_negative_count = sum(1 for row in side_data if row["hardNegative"])
    print(f"Loaded {original_size} source examples.")
    print(f"Sample ratio: {args.sample_ratio} ({len(dataset)} examples kept with seed {args.seed}).")
    print(f"Processed {len(side_data)} examples.")
    print(f"Examples with hard negatives: {hard_negative_count}")
    print(f"All triplets: {args.output}")
    print(f"Train split: {args.train_output} ({len(train_data)})")
    print(f"Valid split: {args.valid_output} ({len(valid_data)})")


if __name__ == "__main__":
    main()
