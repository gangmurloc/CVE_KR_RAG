from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils import load_config, resolve_path, safe_markdown, setup_logging

LOG = setup_logging("evaluate_ai_judge")
SCORE_COLUMNS = [
    "answer_accuracy_0_2",
    "evidence_faithfulness_0_2",
    "citation_correctness_0_2",
    "hallucination_0_2",
    "completeness_0_2",
]
DISPLAY_NAMES = {
    "answer_accuracy_0_2": "Answer Accuracy (0-2)",
    "evidence_faithfulness_0_2": "Evidence Faithfulness (0-2)",
    "citation_correctness_0_2": "Citation Correctness (0-2)",
    "hallucination_0_2": "Hallucination (0-2, lower better)",
    "completeness_0_2": "Completeness (0-2)",
}


def validate(judgments: pd.DataFrame, key: pd.DataFrame) -> None:
    required = {"review_id", "question_type", *SCORE_COLUMNS}
    missing = required - set(judgments.columns)
    if missing:
        raise ValueError(f"AI judge 파일에 필수 열이 없습니다: {sorted(missing)}")
    if judgments["review_id"].duplicated().any():
        raise ValueError("AI judge 파일에 중복 review_id가 있습니다.")
    if set(judgments["review_id"]) != set(key["review_id"]):
        missing_ids = set(key["review_id"]) - set(judgments["review_id"])
        extra_ids = set(judgments["review_id"]) - set(key["review_id"])
        raise ValueError(
            f"review_id 불일치: missing={sorted(missing_ids)[:10]}, "
            f"extra={sorted(extra_ids)[:10]}"
        )
    for column in SCORE_COLUMNS:
        numeric = pd.to_numeric(judgments[column], errors="coerce")
        if numeric.isna().any() or not numeric.isin([0, 1, 2]).all():
            raise ValueError(f"{column}에는 정수 0, 1, 2만 허용됩니다.")


def aggregate(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    summary = frame.groupby(groups, sort=False)[SCORE_COLUMNS].agg(["mean", "std", "count"])
    records: list[dict[str, Any]] = []
    for group_values, row in summary.iterrows():
        values = group_values if isinstance(group_values, tuple) else (group_values,)
        record = dict(zip(groups, values))
        record["N"] = int(row[(SCORE_COLUMNS[0], "count")])
        for column in SCORE_COLUMNS:
            record[DISPLAY_NAMES[column]] = float(row[(column, "mean")])
            record[f"{DISPLAY_NAMES[column]} SD"] = float(row[(column, "std")])
        records.append(record)
    return pd.DataFrame(records)


def evaluate(config: dict[str, Any]) -> list[Path]:
    results = resolve_path(config, "results_dir")
    judgment_path = results / "ai_judge_eval_qwen.csv"
    key_path = results / "human_eval_sample_key_qwen.csv"
    merged_path = results / "ai_judge_scores_qwen.csv"
    if judgment_path.exists() and key_path.exists():
        judgments = pd.read_csv(judgment_path, encoding="utf-8-sig")
        key = pd.read_csv(key_path, encoding="utf-8-sig")
        validate(judgments, key)
        merged = judgments.merge(
            key,
            on="review_id",
            how="inner",
            validate="one_to_one",
            suffixes=("_judge", "_key"),
        )
        mismatch = merged["question_type_judge"] != merged["question_type_key"]
        if mismatch.any():
            raise ValueError("AI judge 파일과 key의 question_type이 일치하지 않습니다.")
        merged = merged.rename(columns={"question_type_key": "question_type"}).drop(
            columns=["question_type_judge"]
        )
    elif merged_path.exists():
        LOG.info("정리된 병합 점수 파일 재사용: %s", merged_path)
        merged = pd.read_csv(merged_path, encoding="utf-8-sig")
        required = {"review_id", "question_type", "method", *SCORE_COLUMNS}
        missing = required - set(merged.columns)
        if missing:
            raise ValueError(f"병합 점수 파일에 필수 열이 없습니다: {sorted(missing)}")
        if merged["review_id"].duplicated().any():
            raise ValueError("병합 점수 파일에 중복 review_id가 있습니다.")
        for column in SCORE_COLUMNS:
            numeric = pd.to_numeric(merged[column], errors="coerce")
            if numeric.isna().any() or not numeric.isin([0, 1, 2]).all():
                raise ValueError(f"{column}에는 정수 0, 1, 2만 허용됩니다.")
    else:
        raise FileNotFoundError(
            "ai_judge_eval_qwen.csv와 key 또는 ai_judge_scores_qwen.csv가 필요합니다."
        )
    merged.to_csv(merged_path, index=False, encoding="utf-8-sig")

    method = aggregate(merged, ["method"])
    overall_values: dict[str, Any] = {"method": "overall", "N": len(merged)}
    for column in SCORE_COLUMNS:
        overall_values[DISPLAY_NAMES[column]] = float(merged[column].mean())
        overall_values[f"{DISPLAY_NAMES[column]} SD"] = float(merged[column].std())
    method = pd.concat([method, pd.DataFrame([overall_values])], ignore_index=True)
    method = method.rename(columns={"method": "Method"})
    method_csv = results / "ai_judge_metrics_qwen.csv"
    method_md = results / "ai_judge_metrics_qwen.md"
    method.to_csv(method_csv, index=False, encoding="utf-8-sig", float_format="%.6f")
    safe_markdown(method, method_md)

    query_type = aggregate(merged, ["question_type", "method"]).rename(columns={
        "question_type": "Query Type",
        "method": "Method",
    })
    query_csv = results / "ai_judge_query_type_metrics_qwen.csv"
    query_md = results / "ai_judge_query_type_metrics_qwen.md"
    query_type.to_csv(query_csv, index=False, encoding="utf-8-sig", float_format="%.6f")
    safe_markdown(query_type, query_md)
    LOG.info("LLM-as-a-Judge 200건 집계 완료: %s", method_csv)
    return [merged_path, method_csv, method_md, query_csv, query_md]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    evaluate(load_config(args.config))


if __name__ == "__main__":
    main()
