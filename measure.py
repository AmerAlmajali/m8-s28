"""Task 3 — Measure recall@5, MRR, and per-query latency."""

from __future__ import annotations

import time
import json
import weaviate
from sentence_transformers import SentenceTransformer

from retrieval_helpers import hybrid_search, CLASS_NAME
from rerank import cross_encoder_rerank

LABELED_SET_PATH = "data/retrieval_eval.jsonl"


def recall_at_k(retrieved: list[str], relevant: list[str], k: int = 5) -> float:
    return float(any(r in retrieved[:k] for r in relevant))


def mrr(retrieved: list[str], relevant: list[str]) -> float:
    for i, doc_id in enumerate(retrieved, 1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def evaluate(client, embedder, labeled_path: str = LABELED_SET_PATH):
    with open(labeled_path, encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f]

    hybrid_recall, hybrid_mrr = [], []
    rerank_recall, rerank_mrr = [], []
    hybrid_latencies, rerank_latencies = [], []
    retrieval_latencies, ce_latencies = [], []

    for pair in pairs:
        query = pair["query"]
        relevant = [pair["gold_doc_id"]]

        # --- Debug ---
        hybrid_50 = hybrid_search(client, query, 50, embedder, alpha=0.5)
        print(f"Gold: {relevant[0]}")
        print(f"Gold in hybrid@50: {relevant[0] in hybrid_50}")
        print(f"Gold position in hybrid@50: {hybrid_50.index(relevant[0]) if relevant[0] in hybrid_50 else 'NOT FOUND'}")
        print()

        # --- Hybrid baseline ---
        t0 = time.perf_counter()
        hybrid_ids = hybrid_search(client, query, 5, embedder, alpha=0.5)
        hybrid_latencies.append((time.perf_counter() - t0) * 1000)
        hybrid_recall.append(recall_at_k(hybrid_ids, relevant))
        hybrid_mrr.append(mrr(hybrid_ids, relevant))

        # --- Rerank pipeline ---
        t0 = time.perf_counter()
        hybrid_candidates = hybrid_search(client, query, 50, embedder, alpha=0.5)
        t1 = time.perf_counter()

        res = (
            client.query
            .get(CLASS_NAME, ["doc_id", "text"])
            .with_where({
                "path": ["doc_id"],
                "operator": "ContainsAny",
                "valueTextArray": hybrid_candidates,
            })
            .with_limit(50)
            .do()
        )
        items = res.get("data", {}).get("Get", {}).get(CLASS_NAME, []) or []
        text_map = {it["doc_id"]: it["text"] for it in items}
        candidates = [
            {"doc_id": did, "text": text_map[did]}
            for did in hybrid_candidates if did in text_map
        ]

        t2 = time.perf_counter()
        reranked_ids = cross_encoder_rerank(query, candidates, k_out=5)
        t3 = time.perf_counter()

        retrieval_latencies.append((t1 - t0) * 1000)
        ce_latencies.append((t3 - t2) * 1000)
        rerank_latencies.append((t3 - t0) * 1000)
        rerank_recall.append(recall_at_k(reranked_ids, relevant))
        rerank_mrr.append(mrr(reranked_ids, relevant))

        print(f"Hybrid@5:  {hybrid_ids}")
        print(f"Reranked@5: {reranked_ids}")
        print(f"Recall hybrid: {hybrid_recall[-1]}  Recall rerank: {rerank_recall[-1]}")
        print("-" * 50)

    print("=" * 50)
    print(f"{'Metric':<30} {'Hybrid':>8} {'Rerank':>8}")
    print("=" * 50)
    print(f"{'recall@5':<30} {sum(hybrid_recall)/len(hybrid_recall):>8.3f} {sum(rerank_recall)/len(rerank_recall):>8.3f}")
    print(f"{'MRR':<30} {sum(hybrid_mrr)/len(hybrid_mrr):>8.3f} {sum(rerank_mrr)/len(rerank_mrr):>8.3f}")
    print(f"{'Avg latency (ms)':<30} {sum(hybrid_latencies)/len(hybrid_latencies):>8.1f} {sum(rerank_latencies)/len(rerank_latencies):>8.1f}")
    print(f"{'  hybrid stage (ms)':<30} {'':>8} {sum(retrieval_latencies)/len(retrieval_latencies):>8.1f}")
    print(f"{'  cross-encoder stage (ms)':<30} {'':>8} {sum(ce_latencies)/len(ce_latencies):>8.1f}")
    print("=" * 50)


if __name__ == "__main__":
    client = weaviate.Client("http://localhost:8080")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    evaluate(client, embedder)