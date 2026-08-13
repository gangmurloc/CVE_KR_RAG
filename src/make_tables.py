from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils import as_list, load_config, read_jsonl, resolve_path, safe_markdown, setup_logging

LOG = setup_logging("make_tables")


def save_table(frame: pd.DataFrame, results: Path, stem: str) -> tuple[Path, Path]:
    csv_path, md_path = results / f"{stem}.csv", results / f"{stem}.md"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig", float_format="%.6f")
    display = frame.copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        if not pd.api.types.is_integer_dtype(display[column]):
            display[column] = display[column].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    safe_markdown(display, md_path)
    return csv_path, md_path


def table1(config: dict[str, Any]) -> pd.DataFrame:
    processed = resolve_path(config, "processed_dir")
    cves = read_jsonl(processed / "selected_cves_100.jsonl")
    qa = read_jsonl(processed / "qa_ko_500.jsonl")
    severity = pd.Series([str(x.get("cvss_base_severity", "")).upper() for x in cves]).value_counts()
    cwes = {cwe for row in cves for cwe in as_list(row.get("cwe_ids"))}
    return pd.DataFrame([{
        "# CVEs": len(cves),
        "# QA pairs": len(qa),
        "Critical": int(severity.get("CRITICAL", 0)),
        "High": int(severity.get("HIGH", 0)),
        "Medium": int(severity.get("MEDIUM", 0)),
        "Low": int(severity.get("LOW", 0)),
        "# KEV CVEs": sum(bool(row.get("is_kev")) for row in cves),
        "# CWE types": len(cwes),
    }])


def table2(results: Path) -> pd.DataFrame | None:
    path = results / "retrieval_metrics.csv"
    if not path.exists():
        return None
    columns = ["Method", "Hit@1", "Recall@5", "Recall@10", "MRR@10", "nDCG@10"]
    return pd.read_csv(path)[columns]


def populated_human_metrics(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    faith = pd.to_numeric(frame.get("evidence_faithfulness_0_2"), errors="coerce")
    hallucination = pd.to_numeric(frame.get("hallucination_0_2"), errors="coerce")
    if not faith.notna().any() and not hallucination.notna().any():
        return None
    frame = frame.assign(
        _faithfulness=faith,
        _hallucination=hallucination,
    )
    return frame.groupby("method", sort=False)[["_faithfulness", "_hallucination"]].mean().reset_index()


def table3(results: Path) -> pd.DataFrame | None:
    path = results / "answer_auto_metrics_qwen.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    columns = [
        "Method", "Parse Success Rate", "Field Accuracy", "Severity Accuracy",
        "Attack Vector Accuracy", "CWE F1", "Citation Accuracy",
    ]
    frame = frame[columns]
    human = populated_human_metrics(results / "human_eval_template_qwen.csv")
    if human is not None:
        human = human.rename(columns={
            "method": "Method",
            "_faithfulness": "Human Faithfulness",
            "_hallucination": "Hallucination Score",
        })
        frame = frame.merge(human, on="Method", how="left")
    return frame


def table4(results: Path) -> pd.DataFrame | None:
    retrieval_path = results / "retrieval_results.jsonl"
    if not retrieval_path.exists():
        return None
    retrieval = pd.DataFrame(read_jsonl(retrieval_path))
    retrieval["MRR@10"] = retrieval["rank_of_gold_cve"].map(
        lambda rank: 1.0 / rank if pd.notna(rank) and 1 <= rank <= 10 else 0.0
    )
    grouped = retrieval.groupby(["question_type", "method"], sort=False).agg(
        **{"Hit@1": ("hit_at_1", "mean"), "Recall@5": ("hit_at_5", "mean"), "MRR@10": ("MRR@10", "mean")}
    ).reset_index().rename(columns={"question_type": "Query Type", "method": "Method"})
    answer_path = results / "answer_auto_scores_qwen.csv"
    if answer_path.exists():
        answers = pd.read_csv(answer_path)
        answer_group = answers.groupby(["question_type", "method"], sort=False).agg(
            **{
                "Field Accuracy": ("field_accuracy", "mean"),
                "Citation Accuracy": ("citation_accuracy", "mean"),
            }
        ).reset_index().rename(columns={"question_type": "Query Type", "method": "Method"})
        grouped = grouped.merge(answer_group, on=["Query Type", "Method"], how="left")
    order = [
        "vulnerability_summary", "affected_product", "attack_condition",
        "severity_reason", "mitigation",
    ]
    grouped["Query Type"] = pd.Categorical(grouped["Query Type"], order, ordered=True)
    return grouped.sort_values(["Query Type", "Method"]).reset_index(drop=True)


def make(config: dict[str, Any]) -> list[tuple[Path, Path]]:
    results = resolve_path(config, "results_dir")
    outputs = [save_table(table1(config), results, "table1_dataset_statistics")]
    for number, builder, stem in (
        (2, table2, "table2_retrieval_performance"),
        (3, table3, "table3_answer_reliability_qwen"),
        (4, table4, "table4_query_type_analysis"),
    ):
        frame = builder(results)
        if frame is None:
            LOG.warning("Table %d 입력 결과가 없어 생성을 건너뜁니다.", number)
        else:
            outputs.append(save_table(frame, results, stem))
    LOG.info("%d개 논문용 표(CSV+Markdown) 생성", len(outputs))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    make(load_config(args.config))


if __name__ == "__main__":
    main()

