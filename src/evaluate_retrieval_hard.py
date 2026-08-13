from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils import load_config, read_jsonl, resolve_path, safe_markdown, setup_logging

LOG = setup_logging("evaluate_retrieval_hard")


def metrics_for_multigold(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Retrieval metrics for hard-split rows whose gold label is a *set* of
    equally-correct CVEs (see build_qa_dataset_hard._attach_gold_sets), not a
    single CVE. Hit@k and MRR@10 are first-relevant-rank metrics (the standard
    IR definitions, unaffected by having multiple relevant documents), but
    Recall@k and nDCG@k are not: reusing the single-gold formulas from
    evaluate_retrieval.metrics_for would silently redefine "Recall" as "any hit"
    and "nDCG" as "first-hit discounted rank", both wrong once a query can have
    more than one correct answer.
    """
    count = len(rows)
    keys = ("Hit@1", "Hit@5", "Hit@10", "Recall@5", "Recall@10", "MRR@10", "nDCG@10")
    if not count:
        return {key: float("nan") for key in keys}

    hit1, hit5, hit10, recall5, recall10, mrr, ndcg = [], [], [], [], [], [], []
    for row in rows:
        gold = set(row["gold_cve_id_set"])
        top10 = row.get("retrieved_cve_ids_top10") or []
        top5 = top10[:5]
        rank = row.get("rank_of_gold_cve")

        hit1.append(bool(rank and rank <= 1))
        hit5.append(bool(rank and rank <= 5))
        hit10.append(bool(rank and rank <= 10))
        mrr.append(1.0 / rank if rank and rank <= 10 else 0.0)

        recall5.append(len(set(top5) & gold) / len(gold) if gold else 0.0)
        recall10.append(len(set(top10) & gold) / len(gold) if gold else 0.0)

        dcg = sum(1.0 / math.log2(i + 1) for i, cve in enumerate(top10, 1) if cve in gold)
        ideal_hits = min(len(gold), 10)
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
        ndcg.append(dcg / idcg if idcg > 0 else 0.0)

    def mean(values: list[float]) -> float:
        return float(sum(values) / len(values))

    return {
        "Hit@1": mean(hit1),
        "Hit@5": mean(hit5),
        "Hit@10": mean(hit10),
        "Recall@5": mean(recall5),
        "Recall@10": mean(recall10),
        "MRR@10": mean(mrr),
        "nDCG@10": mean(ndcg),
    }


def _dedupe_query_weighted(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse rows that share (method, question_type, question_ko) down to one.
    All such rows are, by construction, evaluating the exact same query against
    the exact same retrieved ranking and gold set, so keeping every CVE-instance
    duplicate would let one heavily-shared question (e.g. an attribute_only_hard
    profile shared by 34 CVEs) outweigh a unique one 34-to-1 in the average.
    """
    seen: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["method"], row["question_type"], row["query"])
        seen.setdefault(key, row)
    return list(seen.values())


def _tables(rows: list[dict[str, Any]], results_dir: Path, suffix: str) -> tuple[Path, Path, Path, Path]:
    methods = list(dict.fromkeys(row["method"] for row in rows))

    overall = [
        {"Method": method, **metrics_for_multigold([row for row in rows if row["method"] == method])}
        for method in methods
    ]
    overall_frame = pd.DataFrame(overall)
    overall_csv = results_dir / f"retrieval_metrics_hard{suffix}.csv"
    overall_md = results_dir / f"retrieval_metrics_hard{suffix}.md"
    overall_frame.to_csv(overall_csv, index=False, encoding="utf-8-sig", float_format="%.6f")
    overall_display = overall_frame.copy()
    for column in overall_display.columns[1:]:
        overall_display[column] = overall_display[column].map(lambda x: f"{x:.4f}")
    safe_markdown(overall_display, overall_md)

    question_types = list(dict.fromkeys(row["question_type"] for row in rows))
    by_type = [
        {
            "Method": method,
            "Question Type": qtype,
            **metrics_for_multigold([row for row in rows if row["method"] == method and row["question_type"] == qtype]),
        }
        for method in methods
        for qtype in question_types
    ]
    by_type_frame = pd.DataFrame(by_type)
    by_type_csv = results_dir / f"retrieval_metrics_hard_by_type{suffix}.csv"
    by_type_md = results_dir / f"retrieval_metrics_hard_by_type{suffix}.md"
    by_type_frame.to_csv(by_type_csv, index=False, encoding="utf-8-sig", float_format="%.6f")
    by_type_display = by_type_frame.copy()
    for column in by_type_display.columns[2:]:
        by_type_display[column] = by_type_display[column].map(lambda x: f"{x:.4f}")
    safe_markdown(by_type_display, by_type_md)

    return overall_csv, overall_md, by_type_csv, by_type_md


def evaluate(config: dict[str, Any]) -> dict[str, tuple[Path, Path, Path, Path]]:
    results_dir = resolve_path(config, "results_dir")
    rows = read_jsonl(results_dir / "retrieval_results_hard.jsonl")

    cve_weighted = _tables(rows, results_dir, suffix="")
    query_weighted = _tables(_dedupe_query_weighted(rows), results_dir, suffix="_query_weighted")

    LOG.info(
        "Hard 검색 평가 저장 (CVE-weighted n=%d, query-weighted n=%d): %s",
        len(rows) // max(len({row['method'] for row in rows}), 1),
        len(_dedupe_query_weighted(rows)) // max(len({row['method'] for row in rows}), 1),
        cve_weighted[0],
    )
    return {"cve_weighted": cve_weighted, "query_weighted": query_weighted}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    evaluate(load_config(args.config))


if __name__ == "__main__":
    main()
