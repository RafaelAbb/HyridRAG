"""One-time audit of golden_dataset.json: for every row, ask a cheap LLM check
whether reference_contexts actually supports answering user_input. Flags
candidates for human review — it does NOT edit or discard anything itself.

Catches the kind of bug found manually earlier: a question about running
Uvicorn paired with a reference_context about APIRouter prefixes (RAGAS's
TestsetGenerator occasionally mis-pairs question and source).

Also flags rows whose query_style is MISSPELLED/POOR_GRAMMAR separately —
not necessarily wrong, but a distinct failure mode (typo-robustness, not
retrieval quality) worth a deliberate keep/discard decision rather than
silently dragging down the aggregate context_precision/recall numbers.

Run from the repo root with:
    python -m evals.ragas.audit_golden
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from openai import OpenAI

from evals.ragas._shared import load_golden_dataset
from src.config import settings

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"

AUDIT_PROMPT = """You are auditing a RAG evaluation dataset for mismatched question/context pairs.

QUESTION: {question}

REFERENCE_CONTEXTS (the text(s) this question is supposed to be answerable from —
may be multiple chunks for a multi-hop question; together they should support
answering the question, though no single one needs to on its own):
{context}

Does the REFERENCE_CONTEXTS content, taken together, actually contain information
that answers the QUESTION? Answer with exactly one word: YES or NO."""

STYLE_FLAGS = {"MISSPELLED", "POOR_GRAMMAR"}


def check_mismatch(client: OpenAI, question: str, reference_contexts: list[str]) -> bool:
    context = "\n\n---\n\n".join(reference_contexts)
    completion = client.chat.completions.create(
        model=settings.judgement_model,
        temperature=0,
        max_tokens=5,
        messages=[{"role": "user", "content": AUDIT_PROMPT.format(question=question, context=context)}],
    )
    answer = completion.choices[0].message.content.strip().upper()
    return "NO" in answer


def main() -> None:
    golden = load_golden_dataset(GOLDEN_DATASET_PATH)
    client = OpenAI(api_key=settings.openai_api_key)

    mismatches = []
    style_flags = []

    for i, item in enumerate(golden):
        if check_mismatch(client, item["user_input"], item["reference_contexts"]):
            mismatches.append(i)

        if item.get("query_style") in STYLE_FLAGS:
            style_flags.append(i)

        print(f"[{i+1}/{len(golden)}] checked", end="\r")

    print()
    print(f"\n=== Likely mismatched question/reference pairs ({len(mismatches)}) ===")
    for i in mismatches:
        print(f"  [{i}] {golden[i]['user_input']}")

    print(f"\n=== MISSPELLED / POOR_GRAMMAR style rows ({len(style_flags)}) ===")
    for i in style_flags:
        print(f"  [{i}] ({golden[i]['query_style']}) {golden[i]['user_input']}")

    overlap = set(mismatches) & set(style_flags)
    if overlap:
        print(f"\n=== Overlap (flagged for both reasons): {sorted(overlap)} ===")


if __name__ == "__main__":
    main()
