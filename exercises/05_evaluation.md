# Lesson 5 — Evaluation

**Goal:** prove your system works with numbers, and use those numbers to decide which chunking
strategy actually wins.

**Files:** `eval/golden_dataset.json`, `eval/run_eval.py`
**Doc phase:** Phase 4 (Day 9–11)

> "I built a RAG system" → forgettable.
> "I built a RAG system that hits 94% faithfulness and 88% citation accuracy on a 50-question
> eval suite, and I have the data showing recursive chunking beat semantic by 6 points" → that
> sentence gets you hired. This lesson produces that sentence.

This is the single highest-leverage lesson in the project for your career. Evaluation is the
skill the field guide flags as rare and the one most candidates skip.

---

## 5.1 — The golden dataset

**The hardest part of evaluation isn't the harness — it's the data.** A golden dataset is a
set of questions with known-correct answers, hand-built and trusted.

🛡️ **The cardinal rule — write these by hand. Do NOT generate them with an LLM.** The entire
point is that they're *human-verified ground truth*. If an LLM writes both the test and takes
the test, you're measuring nothing. This is the most common way people fool themselves in ML
eval. (In an interview, *leading* with "I hand-labeled the golden set" signals you understand
that evaluation quality is bounded by data quality — a senior insight.)

**Cover these categories deliberately** (each teaches you something different):
- **Simple lookups** — answer sits in one chunk. Baseline competence.
- **Multi-hop** — answer requires combining *two* documents. Tests whether retrieval pulls
  both. The hard, interesting case.
- **Unanswerable** — the corpus genuinely doesn't contain the answer. Tests your "I don't
  know" path from Lesson 4. 🧠 If your system hallucinates here, that's a *more* important
  finding than getting a lookup right.
- **Ambiguous phrasing** — a question worded differently from the source text. Tests semantic
  retrieval.

Aim for 50+. Design the JSON schema yourself — at minimum each case needs: a stable id, the
question, what the correct answer must contain (or a rubric for subjective ones), expected
source documents, and a difficulty tag. Put a few starter examples in `golden_dataset.json` and
grow it.

🛡️ **Best practice — version the dataset.** When you add hard cases over time, you want to see
"the eval bar moved on this date." It's data-as-code.

**Ask me about…** how to write a *good* multi-hop question, or how to handle subjective
questions where there's no single golden string (hint: rubrics + LLM-as-judge).

**Checkpoint 5.1** — you have 50+ hand-written cases across all four categories, valid JSON.

---

## 5.2 — The eval harness: `run_eval.py`

Run every golden question through your full pipeline and score it on multiple dimensions.
🧠 **Don't collapse quality to one number.** A RAG answer can be correct but uncited, or
cited but incomplete. Measure separately:

- **Correctness** — does the answer contain the expected content? (string match for factual
  cases; LLM-as-judge against a rubric for subjective ones)
- **Faithfulness** — are all claims grounded in retrieved context, or did it drift? (this is
  *the* RAG metric — hallucination rate, inverted)
- **Retrieval relevance** — were the right chunks even retrieved? (if retrieval missed, the
  generator never had a chance — this tells you *which stage* to fix)
- **Citation accuracy** — do the citations actually support their claims? (reuse your Lesson 4
  verifier)

🧠 **Why per-stage metrics matter:** if correctness is low, the separation tells you whether to
fix *retrieval* (right chunks not found) or *generation* (right chunks found, bad answer
written). Without that split you're debugging blind. This diagnostic thinking is the whole
game.

You don't have to build a framework from scratch — tools like **RAGAS** implement faithfulness
and relevance metrics for RAG specifically. Look at it and decide: use it, or hand-roll simple
versions to understand the mechanics first? https://docs.ragas.io/
(My teacherly opinion: hand-roll the simple versions *first* so you understand what RAGAS is
computing, then optionally adopt it. You learn more, and you can explain the metric in an
interview instead of saying "RAGAS does it.")

**Checkpoint 5.2** — `python -m eval.run_eval` runs all cases and prints per-case scores plus
an aggregate summary (avg correctness, faithfulness, citation accuracy).

---

## 5.3 — The chunking bake-off (this is the deliverable)

Now the experiment the whole project was building toward. Run the *same* eval suite three
times — once per chunking strategy (fixed / recursive / semantic) — and produce a comparison
table.

🧠 **This is real ML engineering:** you're running a controlled experiment. Same questions,
same retrieval, same generation — only the chunking variable changes. The winner isn't
obvious in advance, and "it depends on the metric" is a perfectly valid, sophisticated finding
(maybe semantic wins faithfulness but recursive wins latency).

🛡️ **Best practice — let data drive the decision, then write it down.** Your README states
which strategy you shipped *and the numbers that justified it*. That sentence is portfolio
gold and the thing an interviewer will dig into.

**Ask me about…** how to make this a fair comparison (controlling variables), or how to present
the results visually.

**Checkpoint 5.3** — a comparison table (3 strategies × your metrics) with a clear,
data-backed conclusion about which to ship.

---

## What "done" looks like for Lesson 5

- [ ] 50+ hand-written golden cases across four categories, versioned
- [ ] `run_eval.py` scoring correctness / faithfulness / retrieval / citations separately
- [ ] An aggregate summary report
- [ ] A three-way chunking comparison with a documented winner
- [ ] The numbers that become your README headline and interview story

Next: **`06_api_and_polish.md`**
