from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluate_retrieval import metrics_for
from src.utils import load_config, read_jsonl, resolve_path, safe_markdown, setup_logging

LOG = setup_logging("evaluate_retrieval_hard")


def evaluate(config: dict[str, Any]) -> tuple[Path, Path, Path, Path]:
    results_dir = resolve_path(config, "results_dir")
    rows = read_jsonl(results_dir / "retrieval_results_hard.jsonl")
    methods = list(dict.fromkeys(row["method"] for row in rows))

    overall = [{"Method": method, **metrics_for([row for row in rows if row["method"] == method])} for method in methods]
    overall_frame = pd.DataFrame(overall)
    overall_csv = results_dir / "retrieval_metrics_hard.csv"
    overall_md = results_dir / "retrieval_metrics_hard.md"
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
            **metrics_for([row for row in rows if row["method"] == method and row["question_type"] == qtype]),
        }
        for method in methods
        for qtype in question_types
    ]
    by_type_frame = pd.DataFrame(by_type)
    by_type_csv = results_dir / "retrieval_metrics_hard_by_type.csv"
    by_type_md = results_dir / "retrieval_metrics_hard_by_type.md"
    by_type_frame.to_csv(by_type_csv, index=False, encoding="utf-8-sig", float_format="%.6f")
    by_type_display = by_type_frame.copy()
    for column in by_type_display.columns[2:]:
        by_type_display[column] = by_type_display[column].map(lambda x: f"{x:.4f}")
    safe_markdown(by_type_display, by_type_md)

    LOG.info("Hard 검색 평가 저장: %s, %s", overall_csv, by_type_csv)
    return overall_csv, overall_md, by_type_csv, by_type_md


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    evaluate(load_config(args.config))


if __name__ == "__main__":
    main()
