from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from tqdm import tqdm

from src.utils import load_config, normalize_text, resolve_path, setup_logging, write_jsonl

API_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
LOG = setup_logging("fetch_nvd")


def english_description(cve: dict[str, Any]) -> str:
    descriptions = cve.get("descriptions", [])
    preferred = next((x for x in descriptions if x.get("lang") == "en"), None)
    return normalize_text((preferred or (descriptions[0] if descriptions else {})).get("value"))


def extract_cvss(cve: dict[str, Any]) -> dict[str, Any]:
    metrics = cve.get("metrics", {})
    for key, version in (
        ("cvssMetricV31", "3.1"),
        ("cvssMetricV30", "3.0"),
        # The fixed answer schema uses CVSS v3 categorical values. Prefer v3
        # when NVD also publishes v4 for the same CVE.
        ("cvssMetricV40", "4.0"),
        ("cvssMetricV2", "2.0"),
    ):
        candidates = metrics.get(key, [])
        if not candidates:
            continue
        primary = next((x for x in candidates if x.get("type") == "Primary"), candidates[0])
        data = primary.get("cvssData", {})
        return {
            "cvss_version": data.get("version", version),
            "cvss_vector": data.get("vectorString"),
            "cvss_base_score": data.get("baseScore"),
            "cvss_base_severity": data.get("baseSeverity") or primary.get("baseSeverity"),
            "attack_vector": data.get("attackVector"),
            "attack_complexity": data.get("attackComplexity"),
            "privileges_required": data.get("privilegesRequired"),
            "user_interaction": data.get("userInteraction"),
            "scope": data.get("scope"),
            "confidentiality_impact": data.get("confidentialityImpact"),
            "integrity_impact": data.get("integrityImpact"),
            "availability_impact": data.get("availabilityImpact"),
        }
    return {key: None for key in (
        "cvss_version", "cvss_vector", "cvss_base_score", "cvss_base_severity",
        "attack_vector", "attack_complexity", "privileges_required",
        "user_interaction", "scope", "confidentiality_impact",
        "integrity_impact", "availability_impact",
    )}


def walk_cpe_matches(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for node in nodes:
        for match in node.get("cpeMatch", []):
            item = {
                key: match.get(key)
                for key in (
                    "criteria", "vulnerable", "versionStartIncluding", "versionStartExcluding",
                    "versionEndIncluding", "versionEndExcluding",
                )
                if match.get(key) is not None
            }
            matches.append(item)
        matches.extend(walk_cpe_matches(node.get("nodes", [])))
    return matches


def parse_cpe(criteria: str) -> dict[str, str]:
    parts = criteria.split(":")
    return {
        "vendor": parts[3] if len(parts) > 3 else "",
        "product": parts[4] if len(parts) > 4 else "",
        "version": parts[5] if len(parts) > 5 else "",
    }


def flatten_record(vulnerability: dict[str, Any]) -> dict[str, Any]:
    cve = vulnerability["cve"]
    configurations = cve.get("configurations", [])
    cpe_matches = walk_cpe_matches(configurations)
    affected = []
    for match in cpe_matches:
        parsed = parse_cpe(match.get("criteria", ""))
        affected.append({**parsed, **match})
    weaknesses = cve.get("weaknesses", [])
    cwe_ids = sorted({
        desc.get("value")
        for weakness in weaknesses
        for desc in weakness.get("description", [])
        if str(desc.get("value", "")).startswith("CWE-")
    })
    references = [
        {"url": ref.get("url"), "source": ref.get("source"), "tags": ref.get("tags", [])}
        for ref in cve.get("references", [])
    ]
    return {
        "cve_id": cve.get("id"),
        "published_date": cve.get("published"),
        "last_modified_date": cve.get("lastModified"),
        "description": english_description(cve),
        **extract_cvss(cve),
        "cwe_ids": cwe_ids,
        "affected_products": affected,
        "references": references,
        "configurations": configurations,
    }


def write_state(path: Path, state: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def request_page(
    session: requests.Session,
    params: dict[str, Any],
    headers: dict[str, str],
    base_delay: float,
    max_attempts: int = 8,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = session.get(
                API_URL,
                params=params,
                headers=headers,
                timeout=(20, 180),
            )
            if response.status_code in {403, 429, 500, 502, 503, 504}:
                raise requests.HTTPError(
                    f"NVD HTTP {response.status_code}", response=response
                )
            response.raise_for_status()
            payload = response.json()
            page = payload.get("vulnerabilities")
            total = int(payload.get("totalResults", 0))
            if not isinstance(page, list):
                raise ValueError("NVD 응답에 vulnerabilities 배열이 없습니다.")
            if not page and int(params["startIndex"]) < total:
                raise ValueError("전체 건수 이전에 빈 NVD 페이지를 받았습니다.")
            return payload
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt == max_attempts:
                break
            wait = max(base_delay, min(120.0, base_delay * (2 ** (attempt - 1))))
            LOG.warning(
                "NVD 요청 실패 (%d/%d, startIndex=%s): %s; %.1f초 후 재시도",
                attempt, max_attempts, params["startIndex"], exc, wait,
            )
            time.sleep(wait)
    assert last_error is not None
    raise RuntimeError(
        f"NVD 페이지 요청이 {max_attempts}회 실패했습니다 "
        f"(startIndex={params['startIndex']}). 체크포인트는 보존됩니다."
    ) from last_error


def fetch(config: dict[str, Any], force: bool = False) -> Path:
    raw_dir = resolve_path(config, "raw_dir")
    output = raw_dir / "nvd_cves.jsonl"
    metadata_path = raw_dir / "nvd_fetch_metadata.json"
    state_path = raw_dir / "nvd_fetch_state.json"
    pages_dir = raw_dir / "nvd_pages_partial"
    fetch_cfg = config.get("fetch", {}).get("nvd", {})
    if output.exists() and not (force or fetch_cfg.get("force", False)):
        LOG.info("캐시 재사용: %s", output)
        return output
    if force or fetch_cfg.get("force", False):
        output.unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)
        shutil.rmtree(pages_dir, ignore_errors=True)

    load_dotenv(Path(config["_root"]) / ".env")
    api_key = os.getenv("NVD_API_KEY", "").strip()
    headers = {"apiKey": api_key} if api_key else {}
    delay = 0.7 if api_key else 6.0
    page_size = min(int(fetch_cfg.get("results_per_page", 1000)), 2000)
    max_records = fetch_cfg.get("max_records")
    request_signature = {
        "resultsPerPage": page_size,
        "pubStartDate": fetch_cfg.get("start_date"),
        "pubEndDate": fetch_cfg.get("end_date"),
        "maxRecords": int(max_records) if max_records else None,
    }
    state: dict[str, Any] = {}
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("request_signature") != request_signature:
            raise RuntimeError(
                "기존 NVD 체크포인트와 현재 수집 설정이 다릅니다. "
                "새 설정으로 시작하려면 --force를 사용하세요."
            )
    start_index = int(state.get("next_start_index", 0))
    total = state.get("total_results")
    pages_dir.mkdir(parents=True, exist_ok=True)
    params: dict[str, Any] = {"startIndex": start_index, "resultsPerPage": page_size}
    if fetch_cfg.get("start_date"):
        params["pubStartDate"] = fetch_cfg["start_date"]
    if fetch_cfg.get("end_date"):
        params["pubEndDate"] = fetch_cfg["end_date"]

    if start_index:
        LOG.info("NVD 체크포인트에서 재개: %d개 완료", start_index)
    effective_total = min(int(total), int(max_records)) if total is not None and max_records else total
    progress = tqdm(
        total=int(effective_total) if effective_total is not None else None,
        initial=start_index,
        desc="NVD CVEs",
        unit="CVE",
    )
    session = requests.Session()
    try:
        while total is None or start_index < int(total):
            if max_records and start_index >= int(max_records):
                break
            params["startIndex"] = start_index
            payload = request_page(session, params, headers, delay)
            total = int(payload.get("totalResults", 0))
            target_total = min(total, int(max_records)) if max_records else total
            progress.total = target_total
            progress.refresh()
            page = payload.get("vulnerabilities", [])
            if not page:
                break
            parsed = [flatten_record(item) for item in page]
            if max_records:
                parsed = parsed[: max(0, int(max_records) - start_index)]
            page_path = pages_dir / f"page_{start_index:09d}_{len(parsed):05d}.jsonl"
            temporary_page = page_path.with_suffix(".tmp")
            write_jsonl(parsed, temporary_page)
            os.replace(temporary_page, page_path)
            start_index += len(parsed)
            progress.update(len(parsed))
            write_state(state_path, {
                "request_signature": request_signature,
                "next_start_index": start_index,
                "total_results": total,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            })
            if start_index < target_total:
                time.sleep(delay)
    finally:
        session.close()
        progress.close()

    target_total = min(int(total or 0), int(max_records)) if max_records else int(total or 0)
    if start_index < target_total:
        raise RuntimeError(
            f"NVD 수집이 미완료입니다: {start_index}/{target_total}. "
            "같은 명령을 다시 실행하면 이어받습니다."
        )
    temporary_output = output.with_suffix(".tmp")
    with temporary_output.open("wb") as destination:
        for page_path in sorted(
            pages_dir.glob("page_*.jsonl"),
            key=lambda path: int(path.name.split("_")[1]),
        ):
            with page_path.open("rb") as source:
                shutil.copyfileobj(source, destination)
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary_output, output)
    metadata_path.write_text(json.dumps({
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "record_count": start_index,
        "api_key_used": bool(api_key),
        "request_parameters": {k: v for k, v in params.items() if k != "startIndex"},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    state_path.unlink(missing_ok=True)
    shutil.rmtree(pages_dir, ignore_errors=True)
    LOG.info("%d개 CVE 저장: %s", start_index, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    fetch(load_config(args.config), args.force)


if __name__ == "__main__":
    main()
