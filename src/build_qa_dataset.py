from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.utils import load_config, read_jsonl, resolve_path, setup_logging, write_jsonl

LOG = setup_logging("build_qa_dataset")

QUESTION_TEMPLATES = [
    ("vulnerability_summary", "{cve_id}는 어떤 취약점인가?"),
    ("affected_product", "{cve_id}의 영향을 받는 제품이나 버전은 무엇인가?"),
    ("attack_condition", "{cve_id}는 원격 공격이 가능한가? 인증이나 사용자 상호작용이 필요한가?"),
    ("severity_reason", "{cve_id}의 위험도가 {severity}로 평가되는 이유는 무엇인가?"),
    ("mitigation", "{cve_id}에 대한 대응 방법이나 완화 방안은 무엇인가?"),
]


def make_row(cve: dict[str, Any], index: int, question_type: str, template: str) -> dict[str, Any]:
    cve_id = cve["cve_id"]
    severity = str(cve.get("cvss_base_severity") or "UNKNOWN").upper()
    return {
        "qa_id": f"{cve_id}-Q{index}",
        "cve_id": cve_id,
        "question_type": question_type,
        "question_ko": template.format(cve_id=cve_id, severity=severity),
        "gold_description": cve.get("description"),
        "gold_severity": severity,
        "gold_cvss_score": cve.get("cvss_base_score"),
        "gold_attack_vector": cve.get("attack_vector"),
        "gold_attack_complexity": cve.get("attack_complexity"),
        "gold_privileges_required": cve.get("privileges_required"),
        "gold_user_interaction": cve.get("user_interaction"),
        "gold_cwe_ids": cve.get("cwe_ids", []),
        "gold_affected_products": cve.get("affected_products", []),
        "gold_references": cve.get("references", []),
        "is_kev": bool(cve.get("is_kev")),
        "gold_evidence_source": ["nvd"] + (["cisa_kev"] if cve.get("is_kev") else []),
    }


def build(config: dict[str, Any], force: bool = False) -> tuple[Path, Path]:
    processed = resolve_path(config, "processed_dir")
    csv_path, jsonl_path = processed / "qa_ko_500.csv", processed / "qa_ko_500.jsonl"
    if csv_path.exists() and jsonl_path.exists() and not force:
        LOG.info("캐시 재사용: %s", jsonl_path)
        return csv_path, jsonl_path
    input_path = processed / "selected_cves_100.jsonl"
    cves = read_jsonl(input_path)
    expected = int(config["experiment"]["questions_per_cve"])
    if expected != len(QUESTION_TEMPLATES):
        raise ValueError(f"questions_per_cve는 템플릿 수 {len(QUESTION_TEMPLATES)}와 같아야 합니다.")
    rows = [
        make_row(cve, index, qtype, template)
        for cve in tqdm(cves, desc="한국어 QA 생성")
        for index, (qtype, template) in enumerate(QUESTION_TEMPLATES, 1)
    ]
    write_jsonl(rows, jsonl_path)
    frame = pd.DataFrame(rows)
    for column in ("gold_cwe_ids", "gold_affected_products", "gold_references", "gold_evidence_source"):
        frame[column] = frame[column].map(lambda x: json.dumps(x, ensure_ascii=False))
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    LOG.info("%d QA 저장: %s", len(rows), jsonl_path)
    return csv_path, jsonl_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build(load_config(args.config), args.force)


if __name__ == "__main__":
    main()

