# Lesson 3 — Hybrid Retrieval

**Goal:** given a question, pull the 5 most relevant chunks using dense + sparse search fused
together, then sharpened by a reranker.

**Files:** `src/retrieval/dense.py`, `sparse.py`, `fusion.py`
**Doc phase:** Phase 2 (Day 3–6)

This is the lesson that makes recruiters care. "I built RAG" is everyone. "I built *hybrid*
retrieval with reciprocal rank fusion and a cross-encoder reranker, and measured that it beat
dense-only by X%" is a hire signal.

---

## 3.1 — `dense.py`: semantic search

**The flow:** embed the query with the *same* model you used for chunks → ask ChromaDB for the
nearest k vectors → return them with similarity scores.

🧠 **Critical detail:** the query must be embedded with the identical model as the documents.
Embeddings from different models live in different spaces and comparing them is meaningless.
This is a real bug people ship.

Define a `DenseResult` dataclass (id, text, score, source, metadata) so dense and sparse
results have parallel shapes — fusion in 3.3 needs them comparable.

🛡️ **Watch the distance/similarity conversion.** ChromaDB's cosine collection returns a
*distance* (lower = better). You probably want a *similarity* (higher = better) for intuition
and for fusion. Convert consistently and write a comment explaining the formula, because
future-you will second-guess it.

**Checkpoint 3.1** — ask a conceptual question (one whose answer uses *different words* than
the question). Dense retrieval should still find the right chunk. That's the semantic magic.

---

## 3.2 — `sparse.py`: keyword search

BM25 scores chunks by term overlap with the query — pure lexical matching, no embeddings.

🧠 **Build the intuition with a test:** search for an exact token that appears in your
docs — a function name, a code, a rare acronym. Then run the *same* query through dense
search. Often sparse nails the exact-match case while dense drifts to "semantically related
but wrong." *That gap is the entire argument for hybrid search.* Feel it firsthand before you
build fusion — it'll make the next step obvious.

Return `SparseResult` with the same shape as `DenseResult`. Note BM25 scores are unbounded
(not 0–1) — that asymmetry is exactly why you can't just add the two scores together, which
motivates 3.3.

**Ask me about…** how BM25 actually scores (term frequency × inverse document frequency, with
length normalisation) if you want to understand *why* it behaves the way it does.

**Checkpoint 3.2** — the exact-keyword query ranks the chunk containing that keyword at or
near the top.

---

## 3.3 — `fusion.py`: combine + rerank (the payoff)

Two sub-problems: **merge** two ranked lists, then **rerank** the merged set.

### Reciprocal Rank Fusion (RRF)

🧠 **Why not just add the scores?** Dense gives 0–1 cosine; BM25 gives unbounded scores. Adding
them lets BM25 dominate arbitrarily. RRF sidesteps this by ignoring the raw scores entirely
and using only *rank position*:

```
score(chunk) = Σ   weight_list / (k + rank_in_that_list)
            over each list the chunk appears in
```

`k` is a constant (60 is the standard from the original paper) that dampens how much the top
ranks dominate. A chunk that ranks high in *both* lists wins; a chunk that's #1 in one list
but absent from the other still scores respectably. Make the dense/sparse weights configurable
(your config already has `rrf_dense_weight` / `rrf_sparse_weight`).

Read the idea here (short and worth it): the original RRF is a 2-page paper, but any "reciprocal
rank fusion explained" write-up covers it. The formula above is the whole thing.

🛡️ **Best practice:** RRF is a *tunable* component. Exposing the weights and `k` as config
(not magic numbers buried in code) is what lets you run experiments in Lesson 5.

### Cross-encoder reranker

After RRF you have, say, the top 20. A **cross-encoder** reads the query and a chunk
*together* (not as separate vectors) and scores their relevance directly. It's slower than
embedding similarity — which is exactly why you only run it on the top 20, not the whole
corpus. This two-stage pattern (cheap broad retrieval → expensive precise rerank) is how
production retrieval is built.

Use `sentence-transformers` with a model like `cross-encoder/ms-marco-MiniLM-L-6-v2`. Docs:
https://www.sbert.net/docs/cross_encoder/usage/usage.html

🧠 **Think about:** the first time you load this model it downloads weights (~80MB) and runs
locally on CPU. What happens to your latency? Is that acceptable for an offline eval but not a
live API? (Foreshadowing a real production tradeoff — note it in your README.)

🛡️ **Best practice — graceful degradation:** if the reranker model fails to load, fall back to
RRF order rather than crashing. A retrieval pipeline that *always returns something* beats one
that's perfect when it works and dead when it doesn't.

Expose one `hybrid_retrieve(query, index, ...)` that runs the whole pipeline: dense → sparse →
RRF → rerank → top 5. This single function is what Lesson 4 and the API call.

**Ask me about…** why a cross-encoder is more accurate than the bi-encoder embeddings you used
for first-pass retrieval, or how to pick `k` and the fusion weights empirically.

**Checkpoint 3.3** — build a query where dense-only returns a mediocre top result, and confirm
that after fusion + rerank the *better* chunk rises to #1. Keep this example — it's your demo
moment and your README's headline comparison.

---

## What "done" looks like for Lesson 3

- [ ] `dense.py` — query embedding + ChromaDB NN search, distance→similarity handled
- [ ] `sparse.py` — BM25 keyword search, parallel result shape
- [ ] `fusion.py` — weighted RRF + cross-encoder rerank + graceful fallback
- [ ] One `hybrid_retrieve()` entry point returning the final top-k
- [ ] A saved example showing hybrid beating dense-only (your future demo)
- [ ] Tests in `tests/test_retrieval.py`

Next: **`04_generation_and_citations.md`**
