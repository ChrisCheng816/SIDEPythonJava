import argparse
import json
import random
from pathlib import Path

from common import REPO_ROOT


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sample_to_match(source_path: Path, target_path: Path, output_path: Path, seed: int) -> tuple[int, int]:
    source_rows = load_jsonl(source_path)
    target_rows = load_jsonl(target_path)

    target_count = len(target_rows)
    if target_count > len(source_rows):
        raise ValueError(
            f"Target split is larger than source split for {source_path.name}: "
            f"target={target_count}, source={len(source_rows)}"
        )

    rng = random.Random(seed)
    sampled = rng.sample(source_rows, target_count)
    sampled.sort(key=lambda row: row.get("fid", 0))
    write_jsonl(sampled, output_path)
    return len(source_rows), target_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample no-side Base splits down to the exact size of the SIDE-filtered splits."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=REPO_ROOT / "study-3" / "replay-runs" / "2026-04-23-base-hf-side09" / "data" / "hf-no-side-raw",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=REPO_ROOT / "study-3" / "replay-runs" / "2026-04-23-base-hf-side09" / "data" / "hf-with-side-threshold-0_9",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "study-3" / "replay-runs" / "2026-04-23-base-hf-side09" / "data" / "hf-no-side-matched-threshold-0_9",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for idx, split in enumerate(["train", "valid", "test"]):
        source_count, sampled_count = sample_to_match(
            args.source_dir / f"{split}.jsonl",
            args.target_dir / f"{split}.jsonl",
            args.output_dir / f"{split}.jsonl",
            seed=args.seed + idx,
        )
        print(
            f"{split}: sampled {sampled_count} rows from {source_count} "
            f"with seed {args.seed + idx}"
        )

    print(f"[OK] Matched no-side splits written to: {args.output_dir}")


if __name__ == "__main__":
    main()
