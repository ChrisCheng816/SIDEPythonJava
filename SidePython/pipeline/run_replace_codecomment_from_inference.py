import argparse
import csv
from pathlib import Path


def load_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replace codeComment in a 500-row annotation CSV with model inference predictions."
    )
    parser.add_argument("--annotation-csv", type=Path, required=True)
    parser.add_argument("--inference-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--annotation-col", default="codeComment")
    parser.add_argument("--prediction-col", default="raw_predictions")
    args = parser.parse_args()

    annotation_fields, annotation_rows = load_csv(args.annotation_csv)
    inference_fields, inference_rows = load_csv(args.inference_csv)

    if args.annotation_col not in annotation_fields:
        raise ValueError(f"Missing annotation column '{args.annotation_col}' in {args.annotation_csv}")
    if args.prediction_col not in inference_fields:
        raise ValueError(f"Missing prediction column '{args.prediction_col}' in {args.inference_csv}")
    if len(annotation_rows) != len(inference_rows):
        raise ValueError(
            f"Row count mismatch: annotation={len(annotation_rows)} inference={len(inference_rows)}"
        )

    for annotation_row, inference_row in zip(annotation_rows, inference_rows):
        annotation_row[args.annotation_col] = inference_row[args.prediction_col]

    output_csv = args.output_csv or args.annotation_csv
    write_csv(output_csv, annotation_fields, annotation_rows)
    print(f"[OK] Replaced {args.annotation_col} for {len(annotation_rows)} rows: {output_csv}")


if __name__ == "__main__":
    main()
