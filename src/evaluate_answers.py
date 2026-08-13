from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils import as_list, load_config, read_jsonl, resolve_path, safe_markdown, setup_logging

LOG = setup_logging("evaluate_answers")


def norm(value: Any) -> str:
    return str(value or "").strip().upper().replace(" ", "_")


def exact(predicted: Any, gold: Any) -> float:
    return float(bool(norm(gold)) and norm(predicted) == norm(gold))


def numeric_match(predicted: Any, gold: Any, tolerance: float = 0.1) -> float:
    try:
        return float(abs(float(predicted) - float(gold)) <= tolerance)
    except (TypeError, ValueError):
        return 0.0


def cwe_f1(predicted: Any, gold: Any) -> float:
    pred_set = {norm(item) for item in as_list(predicted) if norm(item)}
    gold_set = {norm(item) for item in as_list(gold) if norm(item)}
    if not pred_set and not gold_set:
        return 1.0
    if not pred_set or not gold_set:
        return 0.0
    overlap = len(pred_set & gold_set)
    precision, recall = overlap / len(pred_set), overlap / len(gold_set)
    return 2 * precision * recall / (precision + recall) if overlap else 0.0


def score_row(
    answer: dict[str, Any], qa: dict[str, Any], docs_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    fields = answer.get("extracted_fields") or {}
    citations = as_list(answer.get("cited_doc_ids"))
    citation_correct = any(
        docs_by_id.get(doc_id, {}).get("cve_id") == qa["cve_id"] for doc_id in citations
    )
    values = {
        "severity_accuracy": exact(fields.get("severity"), qa.get("gold_severity")),
        "cvss_accuracy": numeric_match(fields.get("cvss_score"), qa.get("gold_cvss_score")),
        "attack_vector_accuracy": exact(fields.get("attack_vector"), qa.get("gold_attack_vector")),
        "privileges_required_accuracy": exact(
            fields.get("privileges_required"), qa.get("gold_privileges_required")
        ),
        "user_interaction_accuracy": exact(
            fields.get("user_interaction"), qa.get("gold_user_interaction")
        ),
        "cwe_f1": cwe_f1(fields.get("cwe_ids"), qa.get("gold_cwe_ids")),
    }
    return {
        "qa_id": qa["qa_id"],
        "cve_id": qa["cve_id"],
        "question_type": qa["question_type"],
        "method": answer["method"],
        "parse_success": bool(answer.get("parse_success")),
        "generation_success": not bool(answer.get("generation_error")),
        **values,
        "field_accuracy": float(np.mean(list(values.values()))),
        "citation_accuracy": float(citation_correct),
    }


def evaluate(config: dict[str, Any]) -> tuple[Path, Path, Path]:
    results = resolve_path(config, "results_dir")
    answers_path = results / "generated_answers_qwen.jsonl"
    if not answers_path.exists():
        raise FileNotFoundError("generated_answers_qwen.jsonl이 없습니다. 먼저 생성을 실행하세요.")
    answers = read_jsonl(answers_path)
    qa_by_id = {
        row["qa_id"]: row
        for row in read_jsonl(resolve_path(config, "processed_dir") / "qa_ko_500.jsonl")
    }
    docs_by_id = {
        row["doc_id"]: row
        for row in read_jsonl(resolve_path(config, "corpus_dir") / "cve_corpus.jsonl")
    }
    scores = [
        score_row(answer, qa_by_id[answer["qa_id"]], docs_by_id)
        for answer in answers if answer["qa_id"] in qa_by_id
    ]
    score_frame = pd.DataFrame(scores)
    score_frame.to_csv(results / "answer_auto_scores_qwen.csv", index=False, encoding="utf-8-sig")
    metric_columns = [
        "parse_success", "field_accuracy", "severity_accuracy", "cvss_accuracy",
        "attack_vector_accuracy", "privileges_required_accuracy",
        "user_interaction_accuracy", "cwe_f1", "citation_accuracy",
    ]
    summary = score_frame.groupby("method", sort=False)[metric_columns].mean().reset_index()
    summary = summary.rename(columns={
        "method": "Method",
        "parse_success": "Parse Success Rate",
        "field_accuracy": "Field Accuracy",
        "severity_accuracy": "Severity Accuracy",
        "cvss_accuracy": "CVSS Accuracy",
        "attack_vector_accuracy": "Attack Vector Accuracy",
        "privileges_required_accuracy": "Privileges Required Accuracy",
        "user_interaction_accuracy": "User Interaction Accuracy",
        "cwe_f1": "CWE F1",
        "citation_accuracy": "Citation Accuracy",
    })
    csv_path = results / "answer_auto_metrics_qwen.csv"
    md_path = results / "answer_auto_metrics_qwen.md"
    summary.to_csv(csv_path, index=False, encoding="utf-8-sig", float_format="%.6f")
    display = summary.copy()
    for column in display.columns[1:]:
        display[column] = display[column].map(lambda x: f"{x:.4f}")
    safe_markdown(display, md_path)

    answer_by_key = {(row["qa_id"], row["method"]): row for row in answers}
    human = score_frame[["qa_id", "cve_id", "question_type", "method"]].copy()
    human["question_ko"] = human["qa_id"].map(lambda x: qa_by_id[x]["question_ko"])
    human["answer_ko"] = human.apply(
        lambda row: answer_by_key[(row["qa_id"], row["method"])].get("answer_ko", ""), axis=1
    )
    human["context_doc_ids"] = human.apply(
        lambda row: json.dumps(
            answer_by_key[(row["qa_id"], row["method"])].get("context_doc_ids", []),
            ensure_ascii=False,
        ), axis=1,
    )
    rating_columns = (
        "answer_accuracy_0_2", "evidence_faithfulness_0_2",
        "citation_correctness_0_2", "hallucination_0_2", "completeness_0_2",
    )
    for column in rating_columns:
        human[column] = ""
    human["reviewer_note"] = ""
    human_path = results / "human_eval_template_qwen.csv"
    if human_path.exists():
        previous = pd.read_csv(human_path, dtype=str, keep_default_na=False)
        preserved = previous[
            ["qa_id", "method", *rating_columns, "reviewer_note"]
        ].drop_duplicates(["qa_id", "method"], keep="last")
        human = human.drop(columns=[*rating_columns, "reviewer_note"]).merge(
            preserved, on=["qa_id", "method"], how="left"
        )
        for column in (*rating_columns, "reviewer_note"):
            human[column] = human[column].fillna("")
    human.to_csv(human_path, index=False, encoding="utf-8-sig")
    LOG.info("답변 평가 저장: %s", csv_path)
    return csv_path, md_path, human_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    evaluate(load_config(args.config))


if __name__ == "__main__":
    main()
