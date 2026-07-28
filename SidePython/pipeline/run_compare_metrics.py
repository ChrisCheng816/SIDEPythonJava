import argparse
from pathlib import Path


def _tokenize(text: str) -> list[str]:
    return str(text).strip().split()


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4: compare SIDE with other metrics.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--summary-txt", type=Path, default=None)
    parser.add_argument("--target-col", default="target")
    parser.add_argument("--pred-col", default="raw_predictions")
    parser.add_argument("--side-col", default="SIDE_score")
    args = parser.parse_args()

    import pandas as pd
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer
    from sacrebleu.metrics import CHRF
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from tqdm import tqdm

    df = pd.read_csv(args.input_csv)
    if args.target_col not in df.columns or args.pred_col not in df.columns:
        raise ValueError(
            f"Missing required columns. Needed: {args.target_col}, {args.pred_col}. "
            f"Got: {list(df.columns)}"
        )

    smooth = SmoothingFunction().method4
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    chrf = CHRF()

    bleu4_scores = []
    rouge_l_scores = []
    meteor_scores = []
    chrf_scores = []
    tfidf_scores = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing metrics"):
        ref = str(row[args.target_col]).strip().lower()
        pred = str(row[args.pred_col]).strip().lower()

        ref_tokens = _tokenize(ref)
        pred_tokens = _tokenize(pred)

        if not ref_tokens or not pred_tokens:
            bleu4_scores.append(0.0)
            rouge_l_scores.append(0.0)
            meteor_scores.append(0.0)
            chrf_scores.append(0.0)
            tfidf_scores.append(0.0)
            continue

        bleu4_scores.append(
            sentence_bleu([ref_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
        )
        rouge_l_scores.append(rouge.score(ref, pred)["rougeL"].fmeasure)
        meteor_scores.append(meteor_score([ref_tokens], pred_tokens))
        chrf_scores.append(chrf.sentence_score(pred, [ref]).score / 100.0)

        tfidf = TfidfVectorizer()
        mat = tfidf.fit_transform([ref, pred])
        tfidf_scores.append(float(cosine_similarity(mat[0], mat[1])[0][0]))

    df["BLEU4"] = bleu4_scores
    df["ROUGE_L"] = rouge_l_scores
    df["METEOR"] = meteor_scores
    df["ChrF"] = chrf_scores
    df["TFIDF_COS"] = tfidf_scores

    output_csv = args.output_csv or args.input_csv.with_name(args.input_csv.stem + "_with_metrics.csv")
    df.to_csv(output_csv, index=False)

    summary_path = args.summary_txt or output_csv.with_suffix(".summary.txt")
    metric_cols = ["BLEU4", "ROUGE_L", "METEOR", "ChrF", "TFIDF_COS"]

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Average metrics\n")
        for col in metric_cols:
            f.write(f"{col}: {df[col].mean():.6f}\n")

        if args.side_col in df.columns:
            f.write("\nSpearman correlation with SIDE\n")
            for col in metric_cols:
                corr = df[args.side_col].corr(df[col], method="spearman")
                f.write(f"{args.side_col} vs {col}: {corr:.6f}\n")
        else:
            f.write(f"\nColumn '{args.side_col}' not found; skipped SIDE correlation.\n")

    print(f"[OK] Metric table saved: {output_csv}")
    print(f"[OK] Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
