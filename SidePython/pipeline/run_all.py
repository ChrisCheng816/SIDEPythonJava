import argparse
import importlib
from pathlib import Path

from common import REPO_ROOT, run_cmd


def has_module(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def find_lfs_pointers() -> list[Path]:
    pointers: list[Path] = []
    for p in REPO_ROOT.rglob("*.jsonl"):
        if ".git" in p.parts or "__pycache__" in p.parts:
            continue
        try:
            with p.open("r", encoding="utf-8-sig") as f:
                if f.readline().strip() == "version https://git-lfs.github.com/spec/v1":
                    pointers.append(p)
        except Exception:
            pass
    return sorted(pointers)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SidePython staged pipeline with preflight checks.")
    parser.add_argument("--python-bin", default="python3")
    parser.add_argument("--run-text-eval", action="store_true")
    parser.add_argument("--skip-data-prep", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-score", action="store_true")
    parser.add_argument("--skip-compare", action="store_true")
    parser.add_argument(
        "--score-input-csv",
        type=Path,
        default=REPO_ROOT
        / "study-3"
        / "inference-results"
        / "inference-results-no-SIDE"
        / "Base_benchmark_inference_results-test.csv",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "study-2" / "training-sidep" / "models" / "mpnet_triplet_no_hardneg_v2-test",
    )
    parser.add_argument(
        "--score-output-csv",
        type=Path,
        default=REPO_ROOT / "study-3" / "inference-results" / "metrics" / "base_with_side.csv",
    )
    args = parser.parse_args()

    required_modules = {
        "data_prep": ["datasets"],
        "train": ["torch", "sentence_transformers", "tqdm"],
        "score": ["torch", "sentence_transformers", "pandas", "tqdm"],
        "compare": ["pandas", "nltk", "rouge_score", "sacrebleu", "sklearn", "tqdm"],
    }

    missing = {k: [m for m in mods if not has_module(m)] for k, mods in required_modules.items()}

    lfs_pointers = find_lfs_pointers()
    if lfs_pointers:
        print(f"[WARN] Found {len(lfs_pointers)} Git LFS pointer files (.jsonl placeholders).")
        print("       Some study scripts will fail until real LFS data is pulled.")

    if not args.skip_data_prep:
        if missing["data_prep"]:
            print(f"[SKIP] Data prep skipped: missing modules {missing['data_prep']}")
        else:
            run_cmd([args.python_bin, str(REPO_ROOT / "pipeline" / "run_prepare_codexglue_python.py")])
            run_cmd([args.python_bin, str(REPO_ROOT / "pipeline" / "run_data_prep.py")])

    if not args.skip_train:
        if missing["train"]:
            print(f"[SKIP] Train skipped: missing modules {missing['train']}")
        else:
            run_cmd([args.python_bin, str(REPO_ROOT / "pipeline" / "run_train_sidepy.py")])

    if not args.skip_score:
        if missing["score"]:
            print(f"[SKIP] Score skipped: missing modules {missing['score']}")
        else:
            cmd = [
                args.python_bin,
                str(REPO_ROOT / "pipeline" / "run_score_sidepy.py"),
                "--csv-input",
                str(args.score_input_csv),
                "--checkpoint",
                str(args.checkpoint),
                "--csv-output",
                str(args.score_output_csv),
            ]
            if args.run_text_eval:
                cmd.append("--run-text-eval")
            run_cmd(cmd)

    if not args.skip_compare:
        if missing["compare"]:
            print(f"[SKIP] Compare skipped: missing modules {missing['compare']}")
        else:
            run_cmd(
                [
                    args.python_bin,
                    str(REPO_ROOT / "pipeline" / "run_compare_metrics.py"),
                    "--input-csv",
                    str(args.score_output_csv if args.score_output_csv.exists() else args.score_input_csv),
                ]
            )

    print("[OK] run_all finished.")


if __name__ == "__main__":
    main()
