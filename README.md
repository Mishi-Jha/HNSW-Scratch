# HNSW-Scratch

A vector search index built from scratch — no FAISS, no ChromaDB internals — 
just NumPy, distance math, and a graph. Built to understand what's actually 
happening under the hood of the retrieval systems used in RAG applications.

## Why this exists

Most RAG apps (including my own [Study Buddy](https://github.com/Mishi-Jha/Study-AI-Tool) / [rag-eval-harness](https://github.com/Mishi-Jha/rag-eval-harness)) 
call a vector database like ChromaDB as a black box. This project opens that 
box: implementing HNSW (Hierarchical Navigable Small World), the algorithm 
ChromaDB itself uses, entirely from first principles — then benchmarking it 
against both an exact brute-force baseline and ChromaDB itself.

## What's implemented

- **Distance functions** — L2 and cosine distance (`core/distance.py`)
- **Brute-force KNN** — vectorized exact search, the ground-truth baseline (`baseline/brute_force.py`)
- **HNSW graph** — from scratch (`core/graph.py`, `core/node.py`):
  - Random layer assignment via exponential decay
  - Greedy single-layer search (upper-layer descent)
  - `ef`-based multi-candidate search using a two-heap approach
  - Bidirectional neighbor wiring on insert
- **Benchmarking suite** (`bench/`) — recall@k, latency, and a ChromaDB baseline for comparison

## Results

Tested on 50,000 synthetic 16-dimensional vectors, k=10:

| `ef_search` | Recall@10 | Latency |
|---|---|---|
| 5   | 0.500 | 2.376 ms |
| 10  | 0.940 | 2.024 ms |
| 20  | 0.970 | 2.668 ms |
| 50  | 1.000 | 4.376 ms |
| 100 | 1.000 | 5.821 ms |

**Brute-force baseline:** 7.523 ms (exact, but scales linearly with dataset size)
**ChromaDB:** 6.785 ms

At `ef_search=10`, this implementation reaches ~94% recall while running 
faster than both the exact brute-force baseline and ChromaDB itself — 
though ChromaDB's number isn't a perfectly controlled comparison (it 
carries persistence/metadata overhead this implementation doesn't).

## Running it

```bash
pip install -r requirements.txt
pytest -v                          # run all tests
python -m bench.run_recall_sweep   # recall vs ef_search
python -m bench.run_latency_sweep  # latency comparison (brute-force / HNSW / ChromaDB)
```

## What I learned

The hardest part wasn't the algorithm on paper — it was the small stuff 
that breaks silently. A `break` instead of `continue` in the neighbor 
loop would have quietly explored far less of the graph without ever 
throwing an error. Using `.add()` on what was actually a list, not a 
set, would have crashed the very first insert. The negated max-heap 
trick for tracking "worst result so far" (`heapq` only gives you a 
min-heap natively) took a while to actually trust — I only believed it 
once I traced through it by hand with real numbers.

Building the brute-force baseline first, before any HNSW code, turned 
out to matter more than I expected — every piece after that had a ground 
truth to check itself against, which caught real bugs (a broken cosine 
distance formula, a search that silently returned worse results than 
it should have) that would've been much harder to spot inside a graph 
structure with no reference answer.

The recall/latency sweep was the payoff: watching recall climb from 
0.49 to 1.0 as `ef_search` increased, and watching the crossover point 
where HNSW actually became faster than brute-force only appear once 
the dataset grew to tens of thousands of vectors — both matched what 
the HNSW paper predicts, but seeing my own numbers do it was different 
from reading about it.

## Related

- [rag-eval-harness](https://github.com/Mishi-Jha/rag-eval-harness) — evaluation harness this project's benchmarking 
  approach is modeled after