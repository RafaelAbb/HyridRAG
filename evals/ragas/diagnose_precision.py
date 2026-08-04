"""Diagnose why context_precision/context_recall scored 0 on specific questions in
the latest run_eval.py results, without spending any new API calls.

Loads an existing evals/results/<timestamp>.json (already-scored run) plus
golden_dataset.json (for reference_contexts, which run_eval.py's saved results don't
include), and for every flagged question prints the golden reference_contexts and the
pipeline's actual retrieved_contexts side by side in full.

Each context snippet is also resolved back to its source file under
evals/ragas/corpus/*.md (exact substring match on the first 150 normalized characters,
falling back to a fuzzy best-match via difflib if no exact hit) — so it's possible to
tell "wrong file entirely" apart from "right file, different section" at a glance,
instead of re-reading long text blocks to guess.

Run from the repo root with:
    python -m evals.ragas.diagnose_precision [results_file.json]
"""

import difflib
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from evals.ragas._shared import load_golden_dataset

CORPUS_DIR = Path(__file__).parent / "corpus"
GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
DEFAULT_RESULTS_PATH = Path(__file__).parent.parent / "results" / "20260728T195847Z.json"


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load_corpus() -> dict[str, str]:
    return {f.name: normalize(f.read_text(encoding="utf-8")) for f in CORPUS_DIR.glob("*.md")}


def resolve_source(snippet: str, corpus: dict[str, str]) -> tuple[str, float]:
    probe = normalize(snippet)[:150]
    if not probe:
        return "(empty)", 0.0

    for name, content in corpus.items():
        if probe in content:
            return name, 1.0

    # No exact hit (e.g. reference_context was paraphrased/reflowed by the
    # TestsetGenerator) — fall back to whole-text similarity and report the ratio
    # so a weak fuzzy match is visibly distinguishable from a confident exact one.
    norm_snippet = normalize(snippet)
    best_name, best_ratio = "(no match)", 0.0
    for name, content in corpus.items():
        ratio = difflib.SequenceMatcher(None, norm_snippet, content).quick_ratio()
        if ratio > best_ratio:
            best_ratio, best_name = ratio, name
    return best_name, best_ratio


def print_question(item: dict, golden_item: dict, corpus: dict[str, str]) -> None:
    print("=" * 100)
    print(f"QUESTION: {item['question']}")
    print(f"synthesizer_name={golden_item.get('synthesizer_name')}  query_style={golden_item.get('query_style')}")
    scores = item["scores"]
    print(f"scores: {' | '.join(f'{k}={v:.2f}' for k, v in scores.items())}")

    print(f"\nGOLDEN REFERENCE_CONTEXTS ({len(golden_item['reference_contexts'])}):")
    for i, ctx in enumerate(golden_item["reference_contexts"], start=1):
        source, ratio = resolve_source(ctx, corpus)
        print(f"\n  [{i}] source={source} (match={ratio:.2f})")
        print(f"      {ctx}")

    print(f"\nRETRIEVED_CONTEXTS ({len(item['retrieved_contexts'])}):")
    for i, ctx in enumerate(item["retrieved_contexts"], start=1):
        source, ratio = resolve_source(ctx, corpus)
        print(f"\n  [{i}] source={source} (match={ratio:.2f})")
        print(f"      {ctx}")

    print(f"\nGENERATED ANSWER:\n{item['answer']}")
    print(f"\nGOLDEN REFERENCE ANSWER:\n{golden_item['reference']}")
    print()


def main() -> None:
    results_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RESULTS_PATH

    with open(results_path, encoding="utf-8") as f:
        run = json.load(f)

    golden = load_golden_dataset(GOLDEN_DATASET_PATH)
    golden_by_question = {g["user_input"]: g for g in golden}
    corpus = load_corpus()

    zero_precision = [r for r in run["results"] if r["scores"]["context_precision"] == 0]
    zero_recall = [r for r in run["results"] if r["scores"]["context_recall"] == 0]

    print(f"Loaded {results_path.name}: {len(run['results'])} questions")
    print(f"context_precision == 0: {len(zero_precision)}")
    print(f"context_recall == 0:    {len(zero_recall)}")

    print("\n" + "#" * 100)
    print("# ZERO CONTEXT_PRECISION")
    print("#" * 100)
    for item in zero_precision:
        golden_item = golden_by_question.get(item["question"])
        if golden_item is None:
            print(f"\n[!] No golden_dataset.json entry found for question: {item['question']!r}")
            continue
        print_question(item, golden_item, corpus)

    print("\n" + "#" * 100)
    print("# ZERO CONTEXT_RECALL")
    print("#" * 100)
    for item in zero_recall:
        golden_item = golden_by_question.get(item["question"])
        if golden_item is None:
            print(f"\n[!] No golden_dataset.json entry found for question: {item['question']!r}")
            continue
        print_question(item, golden_item, corpus)


if __name__ == "__main__":
    main()
