from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.utils import as_list, load_config, read_jsonl, resolve_path, set_seed, setup_logging, write_jsonl

LOG = setup_logging("select_cves")


def valid(record: dict[str, Any], min_description_length: int) -> bool:
    severity = str(record.get("cvss_base_severity") or "").upper()
    cvss_version = str(record.get("cvss_version") or "")
    return (
        bool(record.get("cve_id"))
        and len(str(record.get("description") or "")) >= min_description_length
        and severity in {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
        and record.get("cvss_base_score") is not None
        # The answer schema's AV/PR/UI labels are defined in CVSS v3.x.
        and cvss_version.startswith("3.")
        and bool(record.get("attack_vector"))
        and bool(record.get("attack_complexity"))
        and bool(record.get("privileges_required"))
        and bool(record.get("user_interaction"))
    )


def merge_kev(nvd: list[dict[str, Any]], kev: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kev_by_id = {item["cve_id"]: item for item in kev}
    merged = []
    for item in nvd:
        kev_item = kev_by_id.get(item["cve_id"], {})
        merged.append({
            **item,
            "is_kev": bool(kev_item),
            "kev_date_added": kev_item.get("kev_date_added"),
            "kev_known_ransomware_campaign_use": kev_item.get("kev_known_ransomware_campaign_use"),
            "kev_required_action": kev_item.get("kev_required_action"),
            "kev_due_date": kev_item.get("kev_due_date"),
            "kev_vendor_project": kev_item.get("kev_vendor_project"),
            "kev_product": kev_item.get("kev_product"),
            "kev_vulnerability_name": kev_item.get("kev_vulnerability_name"),
            "kev_notes": kev_item.get("kev_notes"),
        })
    return merged


def diversity_select(
    candidates: list[dict[str, Any]], count: int, prefer_kev: bool, kev_limit: int,
    selected_kev: int, seed: int,
) -> list[dict[str, Any]]:
    cwe_counts: Counter[str] = Counter()
    chosen: list[dict[str, Any]] = []
    remaining = sorted(candidates, key=lambda x: x["cve_id"])
    # Seeded deterministic jitter prevents CVE chronology from deciding ties.
    import random
    rng = random.Random(seed)
    jitter = {item["cve_id"]: rng.random() for item in remaining}
    while remaining and len(chosen) < count:
        eligible = [
            item for item in remaining
            if not item.get("is_kev") or selected_kev + sum(bool(x.get("is_kev")) for x in chosen) < kev_limit
        ]
        if not eligible:
            eligible = remaining

        def score(item: dict[str, Any]) -> tuple[float, float, str]:
            cwes = as_list(item.get("cwe_ids")) or ["NO_CWE"]
            diversity_penalty = sum(cwe_counts[cwe] for cwe in cwes) / len(cwes)
            kev_bonus = 0.5 if prefer_kev and item.get("is_kev") else 0.0
            return (diversity_penalty - kev_bonus, jitter[item["cve_id"]], item["cve_id"])

        picked = min(eligible, key=score)
        chosen.append(picked)
        cwe_counts.update(as_list(picked.get("cwe_ids")) or ["NO_CWE"])
        remaining.remove(picked)
    return chosen


def select(config: dict[str, Any], force: bool = False) -> tuple[Path, Path]:
    processed = resolve_path(config, "processed_dir")
    csv_path = processed / "selected_cves_100.csv"
    jsonl_path = processed / "selected_cves_100.jsonl"
    if csv_path.exists() and jsonl_path.exists() and not force:
        LOG.info("캐시 재사용: %s", jsonl_path)
        return csv_path, jsonl_path
    raw = resolve_path(config, "raw_dir")
    nvd_path, kev_path = raw / "nvd_cves.jsonl", raw / "cisa_kev.jsonl"
    if not nvd_path.exists() or not kev_path.exists():
        raise FileNotFoundError("먼저 fetch_nvd.py와 fetch_cisa_kev.py를 실행하세요.")
    nvd = read_jsonl(nvd_path)
    kev = read_jsonl(kev_path)
    sampling = config["sampling"]
    candidates = [
        item for item in tqdm(merge_kev(nvd, kev), desc="CVE 필터링")
        if valid(item, int(sampling["min_description_length"]))
    ]
    targets = {k.upper(): int(v) for k, v in sampling["target_severity_distribution"].items()}
    requested = int(config["experiment"]["num_cves"])
    if sum(targets.values()) != requested:
        LOG.warning("severity target 합(%d)이 num_cves(%d)와 다릅니다.", sum(targets.values()), requested)
    kev_limit = max(1, int(requested * float(sampling.get("kev_max_fraction", 0.6))))
    selected: list[dict[str, Any]] = []
    for index, (severity, target) in enumerate(targets.items()):
        pool = [x for x in candidates if str(x["cvss_base_severity"]).upper() == severity]
        items = diversity_select(
            pool, target, bool(sampling.get("prefer_kev", True)), kev_limit,
            sum(bool(x.get("is_kev")) for x in selected),
            int(config["experiment"]["seed"]) + index,
        )
        if len(items) < target:
            LOG.warning("%s 후보 부족: 목표 %d, 선정 %d", severity, target, len(items))
        selected.extend(items)
    if len(selected) < requested:
        used = {x["cve_id"] for x in selected}
        fill = diversity_select(
            [x for x in candidates if x["cve_id"] not in used],
            requested - len(selected), bool(sampling.get("prefer_kev", True)),
            kev_limit, sum(bool(x.get("is_kev")) for x in selected),
            int(config["experiment"]["seed"]) + 99,
        )
        selected.extend(fill)
    selected = selected[:requested]
    if len(selected) < requested:
        raise RuntimeError(f"유효 CVE가 부족합니다: 필요 {requested}, 가능 {len(selected)}")
    write_jsonl(selected, jsonl_path)
    flat = pd.DataFrame(selected)
    for column in ("cwe_ids", "affected_products", "references", "configurations"):
        if column in flat:
            flat[column] = flat[column].map(lambda x: json.dumps(x, ensure_ascii=False))
    flat.to_csv(csv_path, index=False, encoding="utf-8-sig")
    counts = Counter(str(x["cvss_base_severity"]).upper() for x in selected)
    LOG.info("선정 완료: %s, KEV=%d, severity=%s", jsonl_path, sum(x["is_kev"] for x in selected), dict(counts))
    return csv_path, jsonl_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(int(config["experiment"]["seed"]))
    select(config, args.force)


if __name__ == "__main__":
    main()
