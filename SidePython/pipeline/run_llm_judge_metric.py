import argparse
import json
import os
import re
import site
import sys
from pathlib import Path
from importlib.metadata import PackageNotFoundError, version

from llm_judge_prompt import SYSTEM_PROMPT, build_judge_prompt


JUDGE_SCORE_ALIASES = {
    "content_adequacy": ("content_adequacy", "Content Adequacy", "content adequacy", "contentAdequacy"),
    "conciseness": ("conciseness", "Conciseness"),
    "fluency": ("fluency", "Fluency"),
}


def isolate_user_site_packages() -> None:
    user_site = site.getusersitepackages()
    if isinstance(user_site, str):
        user_sites = [user_site]
    else:
        user_sites = list(user_site)
    for path in user_sites:
        while path in sys.path:
            sys.path.remove(path)
    os.environ.setdefault("PYTHONNOUSERSITE", "1")


def clamp_score(value) -> int:
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        match = re.search(r"\b([1-5])\b", str(value))
        if not match:
            raise
        score = int(match.group(1))
    return max(1, min(5, score))


def get_score(payload: dict, aliases: tuple[str, ...]) -> int:
    for alias in aliases:
        if alias in payload:
            return clamp_score(payload[alias])
    raise KeyError(aliases[0])


def parse_judge_output(text: str) -> tuple[float, int, int, int, str]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            payload = None

        if isinstance(payload, dict):
            try:
                score_payload = payload.get("scores", payload)
                component_scores = {
                    name: get_score(score_payload, aliases)
                    for name, aliases in JUDGE_SCORE_ALIASES.items()
                }
                average = sum(component_scores.values()) / len(component_scores)
                reason = str(payload.get("reason", "")).strip()
                return (
                    average,
                    component_scores["content_adequacy"],
                    component_scores["conciseness"],
                    component_scores["fluency"],
                    reason,
                )
            except (KeyError, TypeError, ValueError):
                try:
                    score = clamp_score(payload["score"])
                    reason = str(payload.get("reason", "")).strip()
                    return score, score, score, score, reason
                except (KeyError, TypeError, ValueError):
                    pass

    number_match = re.search(r"\b([1-5])\b", text)
    if number_match:
        score = int(number_match.group(1))
        return score, score, score, score, "Parsed score from non-JSON model output."

    return 1.0, 1, 1, 1, "Failed to parse model output."


def build_model_input(tokenizer, code: str, summary: str):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_judge_prompt(code, summary)},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{SYSTEM_PROMPT}\n\n{messages[1]['content']}\n\nJSON:"


def check_gpt_oss_environment(model_name: str) -> None:
    if model_name != "openai/gpt-oss-20b" and model_name != "openai/gpt-oss-120b":
        return

    try:
        transformers_version = version("transformers")
    except PackageNotFoundError:
        transformers_version = "not installed"

    missing = []
    for package in ("torch", "triton", "kernels", "typing_extensions"):
        try:
            version(package)
        except PackageNotFoundError:
            missing.append(package)

    if transformers_version == "not installed" or tuple(int(part) for part in transformers_version.split(".")[:2]) < (4, 55):
        raise RuntimeError(
            f"{model_name} uses model_type='gpt_oss', but this environment has transformers=={transformers_version}. "
            "Install the gpt-oss runtime dependencies first:\n"
            "  pip install -U transformers accelerate torch triton==3.4 kernels\n"
            "Then rerun the same pipeline command."
        )

    if missing:
        raise RuntimeError(
            f"{model_name} requires additional runtime package(s): {', '.join(missing)}.\n"
            "Install them with:\n"
            "  PYTHONNOUSERSITE=1 python -m pip install --no-user -U transformers accelerate torch triton==3.4 kernels typing_extensions"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Score code-summary pairs with an LLM-as-a-judge metric.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-txt", type=Path, default=None)
    parser.add_argument("--model-name", default="openai/gpt-oss-20b")
    parser.add_argument("--cuda-visible-devices", default="2,3")
    parser.add_argument("--code-col", default="codeFunctions")
    parser.add_argument("--summary-col", default="codeComment")
    parser.add_argument("--side-col", default="SIDE_score")
    parser.add_argument("--score-col", default="LLM_JUDGE_score")
    parser.add_argument("--content-adequacy-col", default="LLM_JUDGE_content_adequacy")
    parser.add_argument("--conciseness-col", default="LLM_JUDGE_conciseness")
    parser.add_argument("--fluency-col", default="LLM_JUDGE_fluency")
    parser.add_argument("--reason-col", default="LLM_JUDGE_reason")
    parser.add_argument("--raw-col", default="LLM_JUDGE_raw_output")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    isolate_user_site_packages()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    import pandas as pd
    from tqdm import tqdm

    check_gpt_oss_environment(args.model_name)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    df = pd.read_csv(args.input_csv)
    missing = [col for col in (args.code_col, args.summary_col) if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s) {missing} in {args.input_csv}. Got: {list(df.columns)}")

    if args.limit is not None:
        df = df.head(args.limit).copy()

    if args.resume and args.output_csv.exists():
        previous = pd.read_csv(args.output_csv)
        if len(previous) == len(df) and args.score_col in previous.columns:
            df = previous

    score_cols = (
        args.score_col,
        args.content_adequacy_col,
        args.conciseness_col,
        args.fluency_col,
    )
    for col in (*score_cols, args.reason_col, args.raw_col):
        if col not in df.columns:
            df[col] = pd.NA

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Running LLM judge"):
        if args.resume and all(pd.notna(row.get(col)) for col in score_cols):
            continue

        code = str(row[args.code_col])
        summary = str(row[args.summary_col])
        prompt = build_model_input(tokenizer, code, summary)
        encoded = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        output_ids = generated[0][encoded["input_ids"].shape[-1] :]
        raw_output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        score, content_adequacy, conciseness, fluency, reason = parse_judge_output(raw_output)
        df.at[idx, args.score_col] = score
        df.at[idx, args.content_adequacy_col] = content_adequacy
        df.at[idx, args.conciseness_col] = conciseness
        df.at[idx, args.fluency_col] = fluency
        df.at[idx, args.reason_col] = reason
        df.at[idx, args.raw_col] = raw_output

        if (idx + 1) % 25 == 0:
            df.to_csv(args.output_csv, index=False)

    df.to_csv(args.output_csv, index=False)

    summary_path = args.summary_txt or args.output_csv.with_suffix(".summary.txt")
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("LLM judge metric\n")
        handle.write(f"Model: {args.model_name}\n")
        handle.write(f"Rows: {len(df)}\n")
        handle.write(f"{args.score_col} average: {pd.to_numeric(df[args.score_col]).mean():.6f}\n")
        handle.write(f"{args.content_adequacy_col} average: {pd.to_numeric(df[args.content_adequacy_col]).mean():.6f}\n")
        handle.write(f"{args.conciseness_col} average: {pd.to_numeric(df[args.conciseness_col]).mean():.6f}\n")
        handle.write(f"{args.fluency_col} average: {pd.to_numeric(df[args.fluency_col]).mean():.6f}\n")
        if args.side_col in df.columns:
            side = pd.to_numeric(df[args.side_col], errors="coerce")
            judge = pd.to_numeric(df[args.score_col], errors="coerce")
            corr = side.corr(judge, method="spearman")
            handle.write("\nSpearman correlation with SIDE\n")
            handle.write(f"{args.side_col} vs {args.score_col}: {corr:.6f}\n")
        else:
            handle.write(f"\nColumn '{args.side_col}' not found; skipped SIDE correlation.\n")

    print(f"[OK] LLM judge CSV saved: {args.output_csv}")
    print(f"[OK] LLM judge summary saved: {summary_path}")


if __name__ == "__main__":
    main()
