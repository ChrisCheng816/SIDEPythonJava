SYSTEM_PROMPT = """You are an expert evaluator for Python code summarization.
Your task is to judge a generated natural-language summary for the given Python code.
Do not compare against a reference summary. Use only the code and the generated summary.
Score Content Adequacy, Conciseness, and Fluency separately on a 1-5 scale.
Return only valid JSON."""


def build_judge_prompt(code: str, summary: str) -> str:
    return f"""Evaluate the generated summary for the Python code below.

Score each dimension from 1 to 5:

Content Adequacy:
1: Incorrect or mostly unrelated to the code.
2: Mentions a small part of the code but misses the main behavior or includes major errors.
3: Partially correct, but incomplete, vague, or contains some misleading details.
4: Mostly correct and useful, with only minor omissions or imprecision.
5: Accurate, faithful, and useful. It captures the main purpose and behavior without needing every minor detail.

Conciseness:
1: Very verbose, redundant, or filled with irrelevant details.
2: Overly long or includes several unnecessary details.
3: Acceptable length but could be more focused.
4: Mostly concise with only minor extra wording.
5: Clear and compact with no unnecessary detail.

Fluency:
1: Hard to understand due to grammar, wording, or structure.
2: Understandable only with effort; awkward or error-prone wording.
3: Mostly understandable, but with noticeable grammar or phrasing issues.
4: Clear and natural with only minor wording issues.
5: Fluent, grammatical, and easy to read.

Important constraints:
- Judge only the generated summary against the code.
- Do not use any reference or ground-truth summary.
- Do not be too strict on Content Adequacy; reward summaries that correctly capture the main purpose and behavior.
- Penalize hallucinated behavior not supported by the code.
- Penalize summaries that are too generic to be useful under Content Adequacy.
- Penalize unnecessary verbosity under Conciseness, not Content Adequacy.
- Penalize grammar and readability issues under Fluency.

Python code:
```python
{code}
```

Generated summary:
{summary}

Return exactly this JSON object and nothing else:
{{"content_adequacy": <integer from 1 to 5>, "conciseness": <integer from 1 to 5>, "fluency": <integer from 1 to 5>, "reason": "<one short sentence>"}}"""
