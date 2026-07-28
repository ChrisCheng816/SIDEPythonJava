import argparse
import collections
import json
import pickle
import sys
import types
from pathlib import Path
from typing import Optional

from common import REPO_ROOT


class _CompatTokenizer:
    def __init__(self):
        self.word_count = collections.Counter()
        self.w2i = {}
        self.i2w = {}
        self.oov_index = None
        self.vocab_size = None
        self.vectors = {}


_tokenizer_module = types.ModuleType("tokenizer")
_tokenizer_module.Tokenizer = _CompatTokenizer
sys.modules.setdefault("tokenizer", _tokenizer_module)


def _clean_tokens(token_ids, tokenizer) -> list[str]:
    tokens = []
    for raw in token_ids:
        token_id = int(raw)
        if token_id == 0:
            continue
        token = tokenizer.i2w.get(token_id)
        if not token or token in {"<s>", "</s>"}:
            continue
        tokens.append(token)
    return tokens


def _load_hf_assets(repo_id: str):
    from huggingface_hub import hf_hub_download
    import h5py

    meta_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="dataset_short.pkl")
    seqs_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="dataset_seqs.h5")

    metadata = pickle.load(open(meta_path, "rb"))
    seqs = h5py.File(seqs_path, "r")
    return metadata, seqs


def export_split(metadata, seqs, split: str, output_path: Path, max_records: Optional[int]) -> int:
    split_key = {"train": "train", "valid": "val", "test": "test"}[split]
    code_ds = seqs[f"dt{split_key}"]
    comment_ds = seqs[f"c{split_key}"]
    locfid = metadata["locfid"][f"dt{split_key}"]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with output_path.open("w", encoding="utf-8") as out:
        for idx in range(code_ds.shape[0]):
            if max_records is not None and written >= max_records:
                break

            code_tokens = _clean_tokens(code_ds[idx], metadata["tdatstok"])
            doc_tokens = _clean_tokens(comment_ds[idx], metadata["comstok"])
            if not code_tokens or not doc_tokens:
                continue

            row = {
                "fid": int(locfid[idx]),
                "code_tokens": code_tokens,
                "docstring_tokens": doc_tokens,
                "docstring": " ".join(doc_tokens),
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    return written


def main():
    parser = argparse.ArgumentParser(description="Export Base train/valid/test JSONL from Hugging Face apcl/funcom-python.")
    parser.add_argument("--repo-id", default="apcl/funcom-python")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "study-3" / "replay-runs" / "2026-04-23-base-hf-side09" / "data" / "hf-no-side",
    )
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()

    metadata, seqs = _load_hf_assets(args.repo_id)

    counts = {}
    for split, filename in [("train", "train.jsonl"), ("valid", "valid.jsonl"), ("test", "test.jsonl")]:
        counts[split] = export_split(metadata, seqs, split, args.output_dir / filename, args.max_records)

    print(f"[OK] Exported HF Base data to: {args.output_dir}")
    for split in ["train", "valid", "test"]:
        print(f"{split}: {counts[split]}")


if __name__ == "__main__":
    main()
