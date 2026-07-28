import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from sentence_transformers import util
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from common import REPO_ROOT


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def compute_side_score(tokenizer, model, code: str, summary: str) -> float:
    encoded_input = tokenizer([summary, code], padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        model_output = model(**encoded_input)

    embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()


def filter_split(
    tokenizer,
    model,
    input_path: Path,
    kept_path: Path,
    rejected_path: Path,
    threshold: float,
) -> tuple[int, int]:
    kept_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    rejected = 0

    with input_path.open("r", encoding="utf-8") as src, kept_path.open("w", encoding="utf-8") as out_ok, rejected_path.open(
        "w", encoding="utf-8"
    ) as out_bad:
        for line in tqdm(src, desc=f"Filtering {input_path.name}"):
            row = json.loads(line)
            score = compute_side_score(tokenizer, model, " ".join(row["code_tokens"]), row["docstring"])
            row["SIDE_score"] = score

            sink = out_ok if score >= threshold else out_bad
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            if score >= threshold:
                kept += 1
            else:
                rejected += 1

    return kept, rejected


def main():
    parser = argparse.ArgumentParser(description="Filter Base JSONL splits with a trained SIDE-py model.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=REPO_ROOT / "study-3" / "replay-runs" / "2026-04-23-base-hf-side09" / "data" / "hf-no-side",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "study-3" / "replay-runs" / "2026-04-23-base-hf-side09" / "data" / "hf-with-side-threshold-0_9",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "study-2" / "training-sidep" / "models" / "mpnet_triplet_no_hardneg_v2-test",
    )
    parser.add_argument("--threshold", type=float, default=0.9)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint))
    model = AutoModel.from_pretrained(str(args.checkpoint)).to(DEVICE)

    for split in ["train", "valid", "test"]:
        kept, rejected = filter_split(
            tokenizer,
            model,
            args.input_dir / f"{split}.jsonl",
            args.output_dir / f"{split}.jsonl",
            args.output_dir / f"{split}_rejected.jsonl",
            args.threshold,
        )
        print(f"{split}: kept={kept} rejected={rejected} threshold={args.threshold}")

    print(f"[OK] SIDE-filtered Base data written to: {args.output_dir}")


if __name__ == "__main__":
    main()
