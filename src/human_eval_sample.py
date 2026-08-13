from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils import load_config, read_jsonl, resolve_path, setup_logging

LOG = setup_logging("human_eval_sample")
SCORE_COLUMNS = [
    "answer_accuracy_0_2",
    "evidence_faithfulness_0_2",
    "citation_correctness_0_2",
    "hallucination_0_2",
    "completeness_0_2",
]


def render_context(doc_ids: list[str], docs: dict[str, dict[str, Any]]) -> str:
    return "\n\n".join(
        f"[doc_id: {doc_id}]\n{docs.get(doc_id, {}).get('text', '[문서 없음]')}"
        for doc_id in doc_ids
    )


def prepare(config: dict[str, Any], per_query_type: int = 10) -> tuple[Path, Path]:
    results = resolve_path(config, "results_dir")
    answers = read_jsonl(results / "generated_answers_qwen.jsonl")
    docs = {
        row["doc_id"]: row
        for row in read_jsonl(resolve_path(config, "corpus_dir") / "cve_corpus.jsonl")
    }
    frame = pd.DataFrame(answers)
    methods = sorted(frame["method"].unique())
    eligible = (
        frame.groupby(["qa_id", "question_type"], as_index=False)
        .agg(method_count=("method", "nunique"), all_parse_success=("parse_success", "all"))
    )
    eligible = eligible[
        (eligible["method_count"] == len(methods)) & eligible["all_parse_success"]
    ]
    seed = int(config["experiment"]["seed"])
    sampled_qa: list[str] = []
    for offset, (query_type, group) in enumerate(
        eligible.groupby("question_type", sort=True)
    ):
        if len(group) < per_query_type:
            raise RuntimeError(
                f"{query_type}의 네 방법 공통 parse-success QA가 부족합니다: "
                f"필요 {per_query_type}, 가능 {len(group)}"
            )
        sampled_qa.extend(
            group.sample(n=per_query_type, random_state=seed + offset)["qa_id"].tolist()
        )
    selected = frame[frame["qa_id"].isin(sampled_qa)].copy()
    selected = selected.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    selected["review_id"] = [f"R{i:03d}" for i in range(1, len(selected) + 1)]

    key_columns = ["review_id", "qa_id", "cve_id", "question_type", "method"]
    key = selected[key_columns].copy()
    key_path = results / "human_eval_sample_key_qwen.csv"
    key.to_csv(key_path, index=False, encoding="utf-8-sig")

    reviewer = pd.DataFrame({
        "review_id": selected["review_id"],
        "question_type": selected["question_type"],
        "question_ko": selected["qa_id"].map(
            {
                row["qa_id"]: row["query"]
                for row in read_jsonl(results / "retrieval_results.jsonl")
            }
        ),
        "answer_ko": selected["answer_ko"],
        "raw_model_output": selected["raw_model_output"],
        "retrieved_contexts": selected.apply(
            lambda row: render_context(row["context_doc_ids"], docs), axis=1
        ),
        "cited_doc_ids": selected["cited_doc_ids"].map(
            lambda value: json.dumps(value, ensure_ascii=False)
        ),
    })
    for column in SCORE_COLUMNS:
        reviewer[column] = ""
    reviewer["reviewer_note"] = ""
    reviewer_path = results / "human_eval_sample_qwen.csv"
    reviewer.to_csv(reviewer_path, index=False, encoding="utf-8-sig")
    LOG.info(
        "블라인드 paired human 평가 표본 %d건 저장: %s",
        len(reviewer), reviewer_path,
    )
    LOG.info("방법 매핑 키 저장(평가자에게 비공개): %s", key_path)
    return reviewer_path, key_path


def merge(config: dict[str, Any]) -> Path:
    results = resolve_path(config, "results_dir")
    reviewer_path = results / "human_eval_sample_qwen.csv"
    key_path = results / "human_eval_sample_key_qwen.csv"
    template_path = results / "human_eval_template_qwen.csv"
    reviewer = pd.read_csv(reviewer_path, dtype=str, keep_default_na=False)
    key = pd.read_csv(key_path, dtype=str, keep_default_na=False)
    completed = reviewer.copy()
    for column in SCORE_COLUMNS:
        completed[column] = pd.to_numeric(completed[column], errors="coerce")
    completed = completed[completed[SCORE_COLUMNS].notna().all(axis=1)].copy()
    if completed.empty:
        raise RuntimeError("완전히 채워진 human 평가 행이 없습니다.")
    invalid = ~completed[SCORE_COLUMNS].isin([0, 1, 2]).all(axis=1)
    if invalid.any():
        bad_ids = completed.loc[invalid, "review_id"].tolist()
        raise ValueError(f"0, 1, 2 이외의 평가 점수가 있습니다: {bad_ids[:10]}")
    scored = completed[
        ["review_id", *SCORE_COLUMNS, "reviewer_note"]
    ].merge(key, on="review_id", how="left", validate="one_to_one")
    template = pd.read_csv(template_path, dtype=str, keep_default_na=False)
    updates = scored.set_index(["qa_id", "method"])
    template = template.set_index(["qa_id", "method"])
    for column in [*SCORE_COLUMNS, "reviewer_note"]:
        template.loc[updates.index, column] = updates[column]
    template.reset_index().to_csv(template_path, index=False, encoding="utf-8-sig")
    LOG.info("%d건의 human 평가를 병합: %s", len(scored), template_path)
    return template_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "merge"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--per-query-type", type=int, default=10)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.action == "prepare":
        prepare(config, args.per_query_type)
    else:
        merge(config)


if __name__ == "__main__":
    main()
