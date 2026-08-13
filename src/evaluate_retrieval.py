from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils import load_config, read_jsonl, resolve_path, safe_markdown, setup_logging

LOG = setup_logging("evaluate_retrieval")


def metrics_for(rows: list[dict[str, Any]]) -> dict[str, float]:
    ranks = [row.get("rank_of_gold_cve") for row in rows]
    count = len(ranks)
    if not count:
        return {key: float("nan") for key in ("Hit@1", "Recall@5", "Recall@10", "MRR@10", "nDCG@10")}
    return {
        "Hit@1": float(np.mean([bool(rank and rank <= 1) for rank in ranks])),
        "Recall@5": float(np.mean([bool(rank and rank <= 5) for rank in ranks])),
        "Recall@10": float(np.mean([bool(rank and rank <= 10) for rank in ranks])),
        "MRR@10": float(np.mean([1.0 / rank if rank and rank <= 10 else 0.0 for rank in ranks])),
        "nDCG@10": float(np.mean([1.0 / np.log2(rank + 1) if rank and rank <= 10 else 0.0 for rank in ranks])),
    }


def evaluate(config: dict[str, Any]) -> tuple[Path, Path]:
    results_dir = resolve_path(config, "results_dir")
    rows = read_jsonl(results_dir / "retrieval_results.jsonl")
    methods = list(dict.fromkeys(row["method"] for row in rows))
    records = [{"Method": method, **metrics_for([row for row in rows if row["method"] == method])} for method in methods]
    frame = pd.DataFrame(records)
    csv_path, md_path = results_dir / "retrieval_metrics.csv", results_dir / "retrieval_metrics.md"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig", float_format="%.6f")
    display = frame.copy()
    for column in display.columns[1:]:
        display[column] = display[column].map(lambda x: f"{x:.4f}")
    safe_markdown(display, md_path)
    LOG.info("검색 평가 저장: %s", csv_path)
    return csv_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    evaluate(load_config(args.config))


if __name__ == "__main__":
    main()
