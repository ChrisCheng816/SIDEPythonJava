import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert the 500-row human annotation CSV into JSONL accepted by Base inference."
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--code-col", default="codeFunctions")
    parser.add_argument("--target-col", default="originalComment")
    parser.add_argument("--id-col", default="id")
    args = parser.parse_args()

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with args.input_csv.open("r", encoding="utf-8", errors="ignore", newline="") as src, args.output_jsonl.open(
        "w", encoding="utf-8"
    ) as dst:
        reader = csv.DictReader(src)
        fields = set(reader.fieldnames or [])
        missing = [field for field in (args.code_col, args.target_col) if field not in fields]
        if missing:
            raise ValueError(f"Missing required column(s) {missing} in {args.input_csv}")

        for row_index, row in enumerate(reader):
            code = (row.get(args.code_col) or "").strip()
            target = (row.get(args.target_col) or "").strip()
            if not code or not target:
                raise ValueError(
                    f"Empty {args.code_col}/{args.target_col} at row {row_index + 2} in {args.input_csv}"
                )
            record = {
                "fid": row.get(args.id_col, row_index),
                "code_tokens": code,
                "docstring_tokens": target,
                "docstring": target,
                "codeFunctions": code,
                "originalComment": target,
            }
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")
            rows_written += 1

    print(f"[OK] Wrote {rows_written} inference JSONL rows: {args.output_jsonl}")


if __name__ == "__main__":
    main()
