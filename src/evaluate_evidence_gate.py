from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils import (
    as_list, load_config, read_jsonl, resolve_path, safe_markdown, setup_logging,
)

LOG = setup_logging("evaluate_evidence_gate")


def explicit_abstention(answer: dict[str, Any]) -> bool:
    parsed = answer.get("parsed_json") or {}
    uncertainty = str(parsed.get("uncertainty") or "").strip()
    text = str(answer.get("answer_ko") or "")
    if not text:
        # On parse failures, inspect only answer_ko rather than the whole raw JSON.
        # Otherwise an "uncertainty": "확인 불가" field can hide a factual,
        # hallucinated answer body.
        raw = str(answer.get("raw_model_output") or "")
        match = re.search(r'"answer_ko"\s*:\s*("(?:\\.|[^"\\])*")', raw)
        if match:
            try:
                text = json.loads(match.group(1))
            except json.JSONDecodeError:
                text = match.group(1).strip('"')
    patterns = (
        "확인 불가",
        "확인할 수 없음",
        "확인할 수 없습니다",
        "제공된 근거만으로는",
        "정보가 없습니다",
        "정보가 없음",
        "찾을 수 없",
        "존재하지 않는",
        "문서에 없",
        "문서 내에 없",
        "발견되지 않",
        "제공되지 않았",
        "제공할 수 없",
        "포함되어 있지 않",
    )
    return any(pattern in text for pattern in patterns) or (
        not text and uncertainty == "확인 불가"
    )


def citation_matches_query(
    answer: dict[str, Any], docs_by_id: dict[str, dict[str, Any]],
) -> bool:
    context_ids = set(as_list(answer.get("context_doc_ids")))
    return any(
        doc_id in context_ids
        and docs_by_id.get(doc_id, {}).get("cve_id") == answer["cve_id"]
        for doc_id in as_list(answer.get("cited_doc_ids"))
    )


def decision_row(
    split: str,
    retrieval: dict[str, Any],
    answer: dict[str, Any],
    docs_by_id: dict[str, dict[str, Any]],
    stress_type: str | None = None,
) -> dict[str, Any]:
    top5 = retrieval.get("retrieved_cve_ids", [])[:5]
    retrieval_id_match = answer["cve_id"] in top5
    parse_success = bool(answer.get("parse_success"))
    citation_match = citation_matches_query(answer, docs_by_id)
    return {
        "split": split,
        "stress_type": stress_type or "in_domain",
        "qa_id": answer["qa_id"],
        "cve_id": answer["cve_id"],
        "question_type": answer.get("question_type"),
        "method": answer["method"],
        "expected_answerable": split == "in_domain",
        "parse_success": parse_success,
        "retrieval_id_match_top5": retrieval_id_match,
        "citation_matches_query_cve": citation_match,
        "model_explicit_abstention": explicit_abstention(answer),
        "accept_ungated": True,
        "accept_retrieval_id_gate": retrieval_id_match,
        "accept_citation_gate": retrieval_id_match and parse_success and citation_match,
    }


def build_decisions(config: dict[str, Any]) -> pd.DataFrame:
    results = resolve_path(config, "results_dir")
    docs_by_id = {
        row["doc_id"]: row
        for row in read_jsonl(resolve_path(config, "corpus_dir") / "cve_corpus.jsonl")
    }
    in_retrieval = {
        (row["qa_id"], row["method"]): row
        for row in read_jsonl(results / "retrieval_results.jsonl")
    }
    stress_retrieval = {
        (row["qa_id"], row["method"]): row
        for row in read_jsonl(results / "stress_retrieval_results.jsonl")
    }
    rows = []
    for answer in read_jsonl(results / "generated_answers_qwen.jsonl"):
        key = (answer["qa_id"], answer["method"])
        rows.append(decision_row(
            "in_domain", in_retrieval[key], answer, docs_by_id
        ))
    for answer in read_jsonl(results / "stress_generated_answers_qwen.jsonl"):
        key = (answer["qa_id"], answer["method"])
        rows.append(decision_row(
            "stress",
            stress_retrieval[key],
            answer,
            docs_by_id,
            answer.get("stress_type"),
        ))
    return pd.DataFrame(rows)


def evaluate(config: dict[str, Any]) -> list[Path]:
    results = resolve_path(config, "results_dir")
    decisions = build_decisions(config)
    decisions_path = results / "evidence_gate_decisions.csv"
    decisions.to_csv(decisions_path, index=False, encoding="utf-8-sig")
    auto = pd.read_csv(results / "answer_auto_scores_qwen.csv")
    methods = list(dict.fromkeys(decisions["method"]))
    gate_columns = {
        "ungated": "accept_ungated",
        "retrieval_id_gate": "accept_retrieval_id_gate",
        "citation_gate": "accept_citation_gate",
    }
    records = []
    for method in methods:
        method_rows = decisions[decisions["method"] == method]
        in_domain = method_rows[method_rows["split"] == "in_domain"]
        stress = method_rows[method_rows["split"] == "stress"]
        method_auto = auto[auto["method"] == method]
        for gate_name, gate_column in gate_columns.items():
            coverage = float(in_domain[gate_column].mean())
            false_accept = float(stress[gate_column].mean())
            accepted_ids = set(in_domain.loc[in_domain[gate_column], "qa_id"])
            accepted_scores = method_auto[method_auto["qa_id"].isin(accepted_ids)]
            field_accuracy = (
                float(accepted_scores["field_accuracy"].mean())
                if len(accepted_scores) else float("nan")
            )
            records.append({
                "Method": method,
                "Gate": gate_name,
                "In-domain Coverage": coverage,
                "False Reject Rate": 1.0 - coverage,
                "Stress Correct Abstention Rate": 1.0 - false_accept,
                "Stress False Accept Rate": false_accept,
                "Decision Balanced Accuracy": (
                    coverage + (1.0 - false_accept)
                ) / 2.0,
                "Accepted Field Accuracy": field_accuracy,
                "Selective Risk": 1.0 - field_accuracy if not np.isnan(field_accuracy) else np.nan,
                "Accepted In-domain N": len(accepted_scores),
                "Stress N": len(stress),
            })
    metrics = pd.DataFrame(records)
    metrics_csv = results / "evidence_gate_metrics.csv"
    metrics_md = results / "evidence_gate_metrics.md"
    metrics.to_csv(metrics_csv, index=False, encoding="utf-8-sig", float_format="%.6f")
    safe_markdown(metrics, metrics_md)

    stress = decisions[decisions["split"] == "stress"]
    stress_summary = stress.groupby(
        ["method", "stress_type"], sort=False
    ).agg(
        N=("qa_id", "count"),
        **{
            "Qwen Explicit Abstention Rate": ("model_explicit_abstention", "mean"),
            "Qwen Parse Success Rate": ("parse_success", "mean"),
            "Retrieval-ID Gate Correct Abstention": (
                "accept_retrieval_id_gate", lambda values: 1.0 - values.mean()
            ),
            "Citation Gate Correct Abstention": (
                "accept_citation_gate", lambda values: 1.0 - values.mean()
            ),
        },
    ).reset_index().rename(columns={
        "method": "Method",
        "stress_type": "Stress Type",
    })
    stress_csv = results / "stress_abstention_metrics.csv"
    stress_md = results / "stress_abstention_metrics.md"
    stress_summary.to_csv(
        stress_csv, index=False, encoding="utf-8-sig", float_format="%.6f"
    )
    safe_markdown(stress_summary, stress_md)
    LOG.info("Evidence Gate 평가 저장: %s", metrics_csv)
    return [decisions_path, metrics_csv, metrics_md, stress_csv, stress_md]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    evaluate(load_config(args.config))


if __name__ == "__main__":
    main()
