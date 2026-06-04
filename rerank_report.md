# Rerank Report — Module 8 Thursday Stretch

> ~250 words. Replace the placeholder text in each section with your analysis.

## Setup

- Hybrid `k_in`: _50_
- Re-ranked `k_out`: _5_
- Cross-encoder model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Hardware (CPU model, RAM, OS): _your environment_

## Metrics Table


| Pipeline | recall@5 | MRR | per-query latency (ms) |
|---|---|---|---|
| Hybrid (lab baseline) | 0.850 | 0.681 | 106.3 |
| Hybrid + cross-encoder rerank | 0.783 | 0.624 | 10,452.6 |

Stage breakdown for rerank pipeline:
- Stage 1 — hybrid retrieve 50 candidates: **97.6 ms**
- Stage 2 — cross-encoder score 50 pairs: **10,299.4 ms**

## When Does Re-Ranking Pay Off?

Re-ranking pays off when the gold document is ranked outside the top 5 by
hybrid but within the top 50. In our 60-pair labeled set, this happened in
2 clear cases: query 6 (gold at hybrid position 11) and query 10 (gold at
hybrid position 8) — both recovered by the cross-encoder into the top 5,
yielding recall = 1.0 where hybrid scored 0.0.

However, re-ranking hurt in 8 queries where hybrid already had the gold doc
at position 0 — the cross-encoder incorrectly demoted it. This happened
because the corpus text fields are long and the cross-encoder scores the
full untruncated text, introducing noise. The net result is a recall drop
of 6.7 points (0.850 → 0.783) and an MRR drop of 5.7 points (0.681 →
0.624). Re-ranking is only worth it when hybrid recall@50 significantly
exceeds hybrid recall@5 — meaning many gold docs sit between position 5
and 50.

## Latency Overhead

The cross-encoder adds **10,299 ms per query** on CPU, a 98x overhead over
the hybrid baseline (106 ms). The overhead is consistent per query because
the cross-encoder always scores exactly `k_in=50` pairs regardless of
corpus size — it does not scale with corpus size, only with `k_in`. Stage 1
(hybrid retrieve) scales with corpus size and accounts for only 97.6 ms
here on a 1,200-doc corpus. Doubling `k_in` to 100 would roughly double
the cross-encoder time to ~20,000 ms on CPU, while hybrid latency would
increase only modestly.

## At What Corpus Size or Query Volume Does It Stop Being Worth It?

On CPU at 10,452 ms/query, the maximum sustainable throughput is roughly
**0.096 QPS** (1 query per 10.4 seconds). Any production system with more
than 1 concurrent user immediately becomes a bottleneck. On GPU the
cross-encoder drops to ~100 ms/query (50–200 ms per the expected outcome),
giving ~10 QPS — acceptable for low-traffic search but not for high-volume
APIs.

The cross-over point on GPU: at **>10 QPS** the cross-encoder layer
becomes the bottleneck and a learned re-ranker or aggressive result caching
(cache top-50 candidates per query hash) is the right next step. On corpus
size, the cross-encoder itself does not slow down with larger corpora since
it always scores a fixed `k_in=50` candidates — but hybrid retrieval slows
as the index grows. Beyond **~100,000 docs**, hybrid latency alone may
exceed 500 ms, making the total pipeline (500 ms + 100 ms GPU rerank)
acceptable only for offline or batch use cases. At that scale, an ANN index
with a learned re-ranker (M9+ territory) or precomputed candidate caching
is the correct architectural choice.