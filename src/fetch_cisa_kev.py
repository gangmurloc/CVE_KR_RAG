from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from src.utils import load_config, resolve_path, setup_logging, write_jsonl

KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
LOG = setup_logging("fetch_cisa_kev")


def normalize(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "cve_id": item.get("cveID"),
        "is_kev": True,
        "kev_date_added": item.get("dateAdded"),
        "kev_known_ransomware_campaign_use": item.get("knownRansomwareCampaignUse"),
        "kev_required_action": item.get("requiredAction"),
        "kev_due_date": item.get("dueDate"),
        "kev_vendor_project": item.get("vendorProject"),
        "kev_product": item.get("product"),
        "kev_vulnerability_name": item.get("vulnerabilityName"),
        "kev_notes": item.get("notes"),
    }


def fetch(config: dict[str, Any], force: bool = False) -> Path:
    raw_dir = resolve_path(config, "raw_dir")
    output = raw_dir / "cisa_kev.jsonl"
    fetch_cfg = config.get("fetch", {}).get("cisa_kev", {})
    if output.exists() and not (force or fetch_cfg.get("force", False)):
        LOG.info("캐시 재사용: %s", output)
        return output
    response = requests.get(KEV_URL, timeout=90)
    response.raise_for_status()
    payload = response.json()
    records = [normalize(item) for item in payload.get("vulnerabilities", [])]
    write_jsonl(records, output)
    (raw_dir / "cisa_kev_metadata.json").write_text(json.dumps({
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "catalog_version": payload.get("catalogVersion"),
        "date_released": payload.get("dateReleased"),
        "count": len(records),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    LOG.info("%d개 KEV 저장: %s", len(records), output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    fetch(load_config(args.config), args.force)


if __name__ == "__main__":
    main()
