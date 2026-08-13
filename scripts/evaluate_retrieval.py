#!/usr/bin/env python3
"""
scripts/evaluate_retrieval.py
==============================
Calculates Hit@1, Hit@3, and MRR metrics for semantic pgvector retrieval
against a dataset of simulated/natural Chat queries (AI2-4 relevance evaluation).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

from retrieval.semantic import PostgresSemanticStore

EVAL_DATASET_PATH = ROOT / "tests" / "fixtures" / "simulated_chat_eval_dataset.json"


def evaluate() -> dict:
    if not EVAL_DATASET_PATH.exists():
        print(f"[ERROR] Evaluation dataset not found at {EVAL_DATASET_PATH}", file=sys.stderr)
        sys.exit(1)

    dataset = json.loads(EVAL_DATASET_PATH.read_text(encoding="utf-8"))

    try:
        store = PostgresSemanticStore.from_env()
    except Exception as exc:
        print(f"[ERROR] Could not connect to PostgreSQL semantic store: {exc}", file=sys.stderr)
        sys.exit(1)

    total = len(dataset)
    hit_1 = 0
    hit_3 = 0
    reciprocal_ranks = []
    failures = []

    print("\n" + "=" * 70)
    print(f"AI2-4 SEMANTIC PGVECTOR RETRIEVAL RELEVANCE EVALUATION ({total} Simulated Queries)")
    print("=" * 70)

    for item in dataset:
        qid = item["id"]
        query = item["query"]
        field_id = item.get("field_id")
        expected_keys = set(item["expected_document_keys"])

        results = store.search_references(
            query=query,
            field_id=field_id,
            top_k=3,
        )

        retrieved_keys = [r.document_key for r in results]

        rank = None
        for idx, key in enumerate(retrieved_keys, start=1):
            if key in expected_keys:
                rank = idx
                break

        if rank == 1:
            hit_1 += 1

        if rank is not None and rank <= 3:
            hit_3 += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
            failures.append({
                "id": qid,
                "query": query,
                "field_id": field_id,
                "expected": list(expected_keys),
                "retrieved": retrieved_keys,
                "rank": rank,
            })

    hit_1_rate = (hit_1 / total) * 100
    hit_3_rate = (hit_3 / total) * 100
    mrr = sum(reciprocal_ranks) / total

    print(f"Hit@1 Rate : {hit_1}/{total} ({hit_1_rate:.1f}%)")
    print(f"Hit@3 Rate : {hit_3}/{total} ({hit_3_rate:.1f}%)")
    print(f"MRR Score  : {mrr:.4f}")
    print("=" * 70)

    if failures:
        print("\nFAILURE ANALYSIS:")
        print("-" * 70)
        for f in failures:
            print(f"ID       : {f['id']}")
            print(f"Query    : {f['query']!r}")
            print(f"Field ID : {f['field_id']}")
            print(f"Expected : {f['expected']}")
            print(f"Retrieved: {f['retrieved']}")
            print(f"Rank     : {f['rank']}")
            print("-" * 70)
    else:
        print("\n[PERFECT MATCH] All simulated Chat-style queries achieved Hit@1 / Hit@3!")

    return {
        "total": total,
        "hit_1": hit_1,
        "hit_3": hit_3,
        "hit_1_rate": hit_1_rate,
        "hit_3_rate": hit_3_rate,
        "mrr": mrr,
        "failures_count": len(failures),
    }


if __name__ == "__main__":
    evaluate()
