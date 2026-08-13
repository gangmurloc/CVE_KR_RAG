from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.build_qa_dataset import QUESTION_TEMPLATES, make_row
from src.select_cves import valid
from src.utils import (
    iter_jsonl, load_config, read_jsonl, resolve_path, setup_logging, write_jsonl,
)

LOG = setup_logging("build_stress_dataset")
HELDOUT_TARGETS = {"CRITICAL": 8, "HIGH": 8, "MEDIUM": 4}


def reservoir_heldout(
    config: dict[str, Any], excluded_ids: set[str],
) -> list[dict[str, Any]]:
    raw_path = resolve_path(config, "raw_dir") / "nvd_cves.jsonl"
    minimum_length = int(config["sampling"]["min_description_length"])
    seed = int(config["experiment"]["seed"]) + 700
    rng = random.Random(seed)
    reservoirs: dict[str, list[dict[str, Any]]] = {
        severity: [] for severity in HELDOUT_TARGETS
    }
    seen = {severity: 0 for severity in HELDOUT_TARGETS}
    for row in tqdm(iter_jsonl(raw_path), desc="Held-out CVE 표본 탐색", unit="CVE"):
        severity = str(row.get("cvss_base_severity") or "").upper()
        if (
            row.get("cve_id") in excluded_ids
            or severity not in HELDOUT_TARGETS
            or not valid(row, minimum_length)
        ):
            continue
        seen[severity] += 1
        target = HELDOUT_TARGETS[severity]
        if len(reservoirs[severity]) < target:
            reservoirs[severity].append(row)
        else:
            replacement = rng.randrange(seen[severity])
            if replacement < target:
                reservoirs[severity][replacement] = row
    missing = {
        severity: target - len(reservoirs[severity])
        for severity, target in HELDOUT_TARGETS.items()
        if len(reservoirs[severity]) < target
    }
    if missing:
        raise RuntimeError(f"Held-out CVE 후보가 부족합니다: {missing}")
    return [
        row
        for severity in ("CRITICAL", "HIGH", "MEDIUM")
        for row in sorted(reservoirs[severity], key=lambda item: item["cve_id"])
    ]


def stress_rows_for_cve(
    cve: dict[str, Any], stress_type: str, suffix: str,
) -> list[dict[str, Any]]:
    rows = []
    for index, (question_type, template) in enumerate(QUESTION_TEMPLATES, 1):
        row = make_row(cve, index, question_type, template)
        row["qa_id"] = f"{cve['cve_id']}-{suffix}-Q{index}"
        row["stress_type"] = stress_type
        row["expected_answerable"] = False
        rows.append(row)
    return rows


def nonexistent_rows() -> list[dict[str, Any]]:
    templates = [
        ("vulnerability_summary", "{cve_id}는 어떤 취약점인가?"),
        ("affected_product", "{cve_id}의 영향을 받는 제품이나 버전은 무엇인가?"),
        ("attack_condition", "{cve_id}는 원격 공격이 가능한가? 인증이나 사용자 상호작용이 필요한가?"),
        ("severity_reason", "{cve_id}의 위험도와 그 평가 이유는 무엇인가?"),
        ("mitigation", "{cve_id}에 대한 대응 방법이나 완화 방안은 무엇인가?"),
    ]
    rows = []
    for number in range(1, 11):
        cve_id = f"CVE-2099-{90000 + number}"
        for index, (question_type, template) in enumerate(templates, 1):
            rows.append({
                "qa_id": f"{cve_id}-NONEXISTENT-Q{index}",
                "cve_id": cve_id,
                "question_type": question_type,
                "question_ko": template.format(cve_id=cve_id),
                "gold_description": None,
                "gold_severity": None,
                "gold_cvss_score": None,
                "gold_attack_vector": None,
                "gold_attack_complexity": None,
                "gold_privileges_required": None,
                "gold_user_interaction": None,
                "gold_cwe_ids": [],
                "gold_affected_products": [],
                "gold_references": [],
                "is_kev": False,
                "gold_evidence_source": [],
                "stress_type": "nonexistent_cve",
                "expected_answerable": False,
            })
    return rows


def build(config: dict[str, Any], force: bool = False) -> tuple[Path, Path]:
    processed = resolve_path(config, "processed_dir")
    jsonl_path = processed / "qa_ko_stress_200.jsonl"
    csv_path = processed / "qa_ko_stress_200.csv"
    if jsonl_path.exists() and csv_path.exists() and not force:
        LOG.info("스트레스 QA 캐시 재사용: %s", jsonl_path)
        return csv_path, jsonl_path
    selected = read_jsonl(processed / "selected_cves_100.jsonl")
    excluded = {row["cve_id"] for row in selected}
    heldout = reservoir_heldout(config, excluded)
    rows = [
        qa
        for cve in heldout
        for qa in stress_rows_for_cve(cve, "heldout_real_cve", "HELDOUT")
    ]
    rows.extend(nonexistent_rows())
    rng = random.Random(int(config["experiment"]["seed"]) + 701)
    evidence_removed = rng.sample(selected, 10)
    rows.extend(
        qa
        for cve in evidence_removed
        for qa in stress_rows_for_cve(cve, "gold_evidence_removed", "REMOVED")
    )
    if len(rows) != 200:
        raise AssertionError(f"스트레스 QA는 200건이어야 합니다: {len(rows)}")
    write_jsonl(rows, jsonl_path)
    frame = pd.DataFrame(rows)
    for column in (
        "gold_cwe_ids", "gold_affected_products", "gold_references",
        "gold_evidence_source",
    ):
        frame[column] = frame[column].map(lambda value: json.dumps(value, ensure_ascii=False))
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    LOG.info(
        "스트레스 QA 200건 저장(heldout=100, nonexistent=50, evidence_removed=50): %s",
        jsonl_path,
    )
    return csv_path, jsonl_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build(load_config(args.config), args.force)


if __name__ == "__main__":
    main()
