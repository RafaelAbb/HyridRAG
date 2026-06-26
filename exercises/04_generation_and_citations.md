# Lesson 4 — Generation & Citations

**Goal:** turn retrieved chunks into a grounded answer that cites its sources, verifies those
citations, and reports a confidence score.

**File:** `src/generation/generator.py`
**Doc phase:** Phase 3 (Day 6–9)

Most candidates stop at "stuff chunks into a prompt, return the LLM's reply." The three things
in this lesson — *grounding*, *citation verification*, *the honest "I don't know"* — are the
production concerns that separate you from them.

---

## 4.1 — Grounded generation

**The core idea:** the LLM must answer *only* from the chunks you give it, and *cite* which
chunk each claim came from. If the chunks don't contain the answer, it must say so rather than
fall back on its training data (which is how hallucinations happen).

This is almost entirely a **prompt engineering** problem, and prompt design *is* an AI
engineering skill. Your system prompt has to instruct the model to:
- answer only from the provided context
- attach a citation marker (`[1]`, `[2]`…) to each factual claim
- explicitly refuse when the context is insufficient
- not infer, speculate, or pad

🧠 **Think about how you present the chunks.** If you number them `[1]…[5]` in the context
block, the model has stable handles to cite. How you format that block directly shapes
citation quality. Experiment.

🛡️ **Best practice — temperature 0.** For a grounded factual system you want determinism, not
creativity. Set temperature to 0 so the same question gives the same answer (also makes your
eval reproducible).

Read OpenAI's chat completions guide if the API shape is new:
https://platform.openai.com/docs/guides/text-generation

**Checkpoint 4.1** — ask something answerable from your docs. The answer is correct and
contains `[N]` markers pointing at real chunks.

---

## 4.2 — The honest "I don't know"

🧠 **Why this is a feature, not a failure:** a system that fabricates a confident answer when
it has no supporting context is *dangerous* in production — it erodes all trust. One that says
"I couldn't find this in the provided documents; you might check X" is genuinely more useful.

Your job: detect the insufficient-context case (the model following your prompt and declining,
and/or a retrieval-confidence threshold below which you don't even try) and return a structured
"no answer" result instead of a fabricated one.

🛡️ **Best practice:** make "no answer" a first-class, structured outcome (a flag on your result
object), not an exception or a magic string. Downstream code and your eval both need to reason
about it cleanly.

**Checkpoint 4.2** — ask something your docs *cannot* answer. The system declines honestly
instead of inventing an answer. (This is also a test case in Lesson 5.)

---

## 4.3 — Citation verification

Here's the subtlety: an LLM can *write* `[2]` after a claim without chunk 2 actually
supporting it. The citation looks right but is wrong. Most RAG systems never check.

**The technique — LLM-as-judge:** after generation, parse out each (claim, cited-chunk) pair
and ask a *second*, cheaper model call: "Does this chunk actually support this claim?
SUPPORTED / NOT SUPPORTED." Flag the unsupported ones.

🧠 **Think about:** you're using an LLM to check an LLM. What are the failure modes? (The judge
can be wrong too.) Why is it still valuable despite that? (It catches the *obvious*
fabrications, which are the majority, cheaply.) This kind of layered-imperfect-checks thinking
is exactly the "evaluation mindset" the job market is screaming for.

🛡️ **Best practice — use a cheaper model for the judge.** Verification runs once per citation;
`gpt-4o-mini` is plenty and keeps cost sane. Matching model power to task value is a real
engineering instinct.

**Ask me about…** the LLM-as-judge pattern more broadly — it reappears in your eval (Lesson 5)
and in Project 13 and Project 15. It's worth understanding deeply now.

**Checkpoint 4.3** — deliberately construct a case where the model might over-cite, and confirm
your verifier flags the unsupported citation.

---

## 4.4 — Confidence score

Combine your signals into one number (0–1) you can show the user and threshold on:
- **retrieval confidence** — how relevant were the top chunks? (your rerank/RRF scores)
- **citation coverage** — what fraction of claims have *verified* citations?
- **the no-answer flag** — forces confidence to 0

🧠 **Design choice:** how do you weight these? There's no single right answer — document your
reasoning in a comment. A defensible, explained heuristic beats a magic formula. In an
interview, "I weighted citation accuracy and retrieval relevance equally because…" is a great
answer; "it's 0.5 times something" is not.

**Checkpoint 4.4** — answers grounded in strong, verified chunks score high; the no-answer case
scores 0; a weakly-supported answer scores in between. The number behaves sensibly.

---

## What "done" looks like for Lesson 4

- [ ] Grounded generation with inline `[N]` citations, temperature 0
- [ ] Structured, honest "no answer" path
- [ ] Citation verification via a cheap LLM-as-judge pass
- [ ] A composite confidence score with documented weighting
- [ ] One `generate_answer(query, chunks)` entry point returning a rich result object

**Milestone:** you now have a *complete* RAG pipeline end-to-end — ingest to grounded,
cited, scored answer. Everything after this is proving it works (Lesson 5) and serving it
(Lesson 6).

Next: **`05_evaluation.md`**
