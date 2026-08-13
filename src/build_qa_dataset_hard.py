from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.utils import load_config, read_jsonl, resolve_path, setup_logging, write_jsonl

LOG = setup_logging("build_qa_dataset_hard")

AV_PHRASE = {
    "NETWORK": "네트워크를 통해 원격으로",
    "ADJACENT_NETWORK": "인접 네트워크를 통해",
    "LOCAL": "로컬 접근으로",
    "PHYSICAL": "물리적 접근으로",
}
PR_PHRASE = {
    "NONE": "인증 없이",
    "LOW": "낮은 권한의 인증만으로",
    "HIGH": "높은 권한의 인증으로",
}
UI_PHRASE = {
    "NONE": "사용자 상호작용 없이",
    "REQUIRED": "사용자 상호작용을 유도하여",
}


def _display_name(raw: str | None) -> str:
    if not raw:
        return ""
    return " ".join(part.capitalize() for part in re.split(r"[_\-]+", raw) if part)


def _impact_phrase(cve: dict[str, Any]) -> str:
    labels = {
        "기밀성": cve.get("confidentiality_impact"),
        "무결성": cve.get("integrity_impact"),
        "가용성": cve.get("availability_impact"),
    }
    high = [name for name, level in labels.items() if level == "HIGH"]
    low = [name for name, level in labels.items() if level == "LOW"]
    parts = []
    if high:
        parts.append("·".join(high) + "에 심각한")
    if low:
        parts.append("·".join(low) + "에 제한적인")
    if not parts:
        return "시스템에 뚜렷한 영향을 주지 않는"
    return " ".join(parts) + " 영향을 주는"


def _matched_products(cve: dict[str, Any]) -> list[dict[str, str]]:
    description = (cve.get("description") or "").lower()
    products = cve.get("affected_products") or []
    matched, seen = [], set()
    for product in products:
        key = (product.get("vendor"), product.get("product"))
        if key in seen:
            continue
        name = (product.get("product") or "").replace("_", " ").lower()
        if name and name in description:
            matched.append(product)
            seen.add(key)
    if not matched and products:
        first = products[0]
        matched.append(first)
        seen.add((first.get("vendor"), first.get("product")))
    return matched[:2]


def _product_version_phrase(matched: list[dict[str, str]]) -> str:
    parts = []
    for product in matched:
        vendor = _display_name(product.get("vendor"))
        name = _display_name(product.get("product"))
        parts.append(f"{vendor} {name}".strip())
    label = ", ".join(dict.fromkeys(parts)) or "해당 소프트웨어"
    versions = ", ".join(
        dict.fromkeys(p.get("version", "") for p in matched if p.get("version") and p.get("version") != "*")
    )
    if versions:
        return f"{label} ({versions} 버전)"
    return label


def build_questions(cve: dict[str, Any]) -> list[tuple[str, str]]:
    matched = _matched_products(cve)
    product_label = _product_version_phrase(matched)
    av = AV_PHRASE.get(cve.get("attack_vector") or "", "알려지지 않은 경로로")
    pr = PR_PHRASE.get(cve.get("privileges_required") or "", "")
    ui = UI_PHRASE.get(cve.get("user_interaction") or "", "")
    impact = _impact_phrase(cve)
    severity = str(cve.get("cvss_base_severity") or "UNKNOWN").upper()
    score = cve.get("cvss_base_score")

    condition = ", ".join(part for part in (av, pr, ui) if part)

    questions = [
        ("product_version_hard", f"{product_label}에 존재하는 보안 취약점은 무엇인가?"),
        (
            "severity_scenario_hard",
            f"{product_label}에서 발견된 CVSS 기본점수 {score}({severity} 등급)의 취약점으로, "
            f"{impact} 것으로 평가된다. 이 취약점은 무엇인가?",
        ),
        (
            "attack_condition_hard",
            f"{product_label}에서 {condition} 악용될 수 있는 취약점은 무엇인가?",
        ),
        (
            "attribute_only_hard",
            f"{condition} 악용 가능하며 {impact} 취약점은 무엇인가?",
        ),
    ]
    return questions


def make_rows(cve: dict[str, Any], index_start: int) -> list[dict[str, Any]]:
    cve_id = cve["cve_id"]
    rows = []
    for offset, (qtype, question) in enumerate(build_questions(cve)):
        rows.append(
            {
                "qa_id": f"{cve_id}-H{index_start + offset}",
                "cve_id": cve_id,
                "question_type": qtype,
                "question_ko": question,
                "gold_description": cve.get("description"),
                "gold_severity": str(cve.get("cvss_base_severity") or "UNKNOWN").upper(),
                "gold_cvss_score": cve.get("cvss_base_score"),
                "gold_attack_vector": cve.get("attack_vector"),
                "gold_attack_complexity": cve.get("attack_complexity"),
                "gold_privileges_required": cve.get("privileges_required"),
                "gold_user_interaction": cve.get("user_interaction"),
                "gold_cwe_ids": cve.get("cwe_ids", []),
                "gold_affected_products": cve.get("affected_products", []),
                "is_kev": bool(cve.get("is_kev")),
                "contains_cve_id": False,
            }
        )
    return rows


def _attach_gold_sets(rows: list[dict[str, Any]]) -> None:
    """Some hard templates strip enough information that more than one CVE in the
    corpus can produce the exact same question text (e.g. attribute_only_hard for
    the common NETWORK/NONE/NONE profile). Any such CVE is an equally valid answer,
    so the gold label for retrieval evaluation must be the *set* of CVEs whose
    deterministic template output is identical to this question, not just the one
    CVE that happened to generate it. Scoring against a single CVE would count a
    semantically correct retrieval as a miss whenever a same-text sibling exists.
    """
    groups: dict[tuple[str, str], list[str]] = {}
    for row in rows:
        key = (row["question_type"], row["question_ko"])
        groups.setdefault(key, []).append(row["cve_id"])
    for row in rows:
        key = (row["question_type"], row["question_ko"])
        gold_set = sorted(set(groups[key]))
        row["gold_cve_id_set"] = gold_set
        row["gold_set_size"] = len(gold_set)


def build(config: dict[str, Any], force: bool = False) -> tuple[Path, Path]:
    processed = resolve_path(config, "processed_dir")
    csv_path = processed / "qa_ko_hard_400.csv"
    jsonl_path = processed / "qa_ko_hard_400.jsonl"
    if csv_path.exists() and jsonl_path.exists() and not force:
        LOG.info("캐시 재사용: %s", jsonl_path)
        return csv_path, jsonl_path
    input_path = processed / "selected_cves_100.jsonl"
    cves = read_jsonl(input_path)

    rows = []
    for cve in tqdm(cves, desc="한국어 hard QA 생성"):
        rows.extend(make_rows(cve, 1))
    for row in rows:
        assert not re.search(r"CVE-\d{4}-\d{4,}", row["question_ko"], re.IGNORECASE), row["question_ko"]
    _attach_gold_sets(rows)

    write_jsonl(rows, jsonl_path)
    frame = pd.DataFrame(rows)
    for column in ("gold_cwe_ids", "gold_affected_products", "gold_cve_id_set"):
        frame[column] = frame[column].map(lambda x: json.dumps(x, ensure_ascii=False))
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    ambiguous = sum(1 for row in rows if row["gold_set_size"] > 1)
    LOG.info("%d hard QA 저장 (gold set 크기 > 1: %d개): %s", len(rows), ambiguous, jsonl_path)
    return csv_path, jsonl_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build(load_config(args.config), args.force)


if __name__ == "__main__":
    main()
