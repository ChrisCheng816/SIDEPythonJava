import argparse

import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import util
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def main():
    parser = argparse.ArgumentParser(description="Compute SIDE scores for code-summary pairs.")
    parser.add_argument("--csv-input", required=True, help="Input CSV path.")
    parser.add_argument("--csv-output", default=None, help="Output CSV path. Defaults to input path.")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint folder path.")
    parser.add_argument("--code-col", default="target", help="Code column name.")
    parser.add_argument("--summary-col", default="raw_predictions", help="Summary column name.")
    parser.add_argument("--score-col", default="SIDE_score", help="Output score column name.")
    args = parser.parse_args()

    csv_output = args.csv_output if args.csv_output else args.csv_input

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModel.from_pretrained(args.checkpoint).to(DEVICE)

    df = pd.read_csv(args.csv_input)
    if args.code_col not in df.columns:
        raise ValueError(f"Missing code column: {args.code_col}")
    summary_col = args.summary_col
    if summary_col not in df.columns:
        fallback_cols = ["raw_predictions", "summary_postprocessed", "summary"]
        for col in fallback_cols:
            if col in df.columns:
                summary_col = col
                print(f"Summary column '{args.summary_col}' not found. Fallback to '{summary_col}'.")
                break
        else:
            raise ValueError(
                f"Missing summary column: {args.summary_col}. "
                f"Available columns: {list(df.columns)}"
            )

    side_scores = []
    print("Computing SIDE scores for each instance...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        code = str(row[args.code_col])
        summary = str(row[summary_col])
        pair = [summary, code]

        encoded_input = tokenizer(pair, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            model_output = model(**encoded_input)

        embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        sim_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
        side_scores.append(sim_score)

    df[args.score_col] = side_scores
    df.to_csv(csv_output, index=False)

    average_score = sum(side_scores) / len(side_scores)
    print(f"Average SIDE score across all instances: {average_score:.4f}")
    print(f"Updated CSV saved to: {csv_output}")


if __name__ == "__main__":
    main()
