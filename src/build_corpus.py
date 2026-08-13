from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.utils import as_list, load_config, normalize_text, read_jsonl, resolve_path, setup_logging, write_jsonl

LOG = setup_logging("build_corpus")


def product_text(products: Any) -> str:
    rendered = []
    for item in as_list(products):
        if not isinstance(item, dict):
            rendered.append(str(item))
            continue
        value = f"{item.get('vendor', '')}/{item.get('product', '')} {item.get('version', '')}".strip()
        bounds = [
            f"{key}={item[key]}" for key in (
                "versionStartIncluding", "versionStartExcluding",
                "versionEndIncluding", "versionEndExcluding",
            ) if item.get(key)
        ]
        rendered.append(value + (f" ({', '.join(bounds)})" if bounds else ""))
    return "; ".join(dict.fromkeys(filter(None, rendered)))


def reference_text(references: Any) -> str:
    parts = []
    for ref in as_list(references):
        if isinstance(ref, dict):
            tags = ", ".join(ref.get("tags", []))
            parts.append(f"{ref.get('url', '')}" + (f" [{tags}]" if tags else ""))
        else:
            parts.append(str(ref))
    return "; ".join(parts)


def build_doc(cve: dict[str, Any]) -> dict[str, Any]:
    cve_id = cve["cve_id"]
    cvss = (
        f"CVSS {cve.get('cvss_version')}: {cve.get('cvss_base_score')} "
        f"{cve.get('cvss_base_severity')}; vector={cve.get('cvss_vector')}; "
        f"attack_vector={cve.get('attack_vector')}; attack_complexity={cve.get('attack_complexity')}; "
        f"privileges_required={cve.get('privileges_required')}; "
        f"user_interaction={cve.get('user_interaction')}; scope={cve.get('scope')}."
    )
    kev = ""
    if cve.get("is_kev"):
        kev = (
            f"CISA KEV: date_added={cve.get('kev_date_added')}; "
            f"known_ransomware_use={cve.get('kev_known_ransomware_campaign_use')}; "
            f"required_action={cve.get('kev_required_action')}; due_date={cve.get('kev_due_date')}."
        )
    text = "\n".join(filter(None, [
        f"CVE ID: {cve_id}",
        f"Description: {normalize_text(cve.get('description'))}",
        cvss,
        f"CWE: {', '.join(as_list(cve.get('cwe_ids'))) or 'not specified'}",
        f"Affected products/versions: {product_text(cve.get('affected_products')) or 'not specified'}",
        kev,
        f"References and possible mitigation sources: {reference_text(cve.get('references')) or 'not specified'}",
    ]))
    return {
        "doc_id": f"{cve_id}-nvd-kev",
        "cve_id": cve_id,
        "source_type": "cisa_kev" if cve.get("is_kev") else "nvd",
        "text": text,
        "severity": cve.get("cvss_base_severity"),
        "cwe_ids": cve.get("cwe_ids", []),
        "is_kev": bool(cve.get("is_kev")),
    }


def build(config: dict[str, Any], force: bool = False) -> Path:
    output = resolve_path(config, "corpus_dir") / "cve_corpus.jsonl"
    if output.exists() and not force:
        LOG.info("캐시 재사용: %s", output)
        return output
    source = resolve_path(config, "processed_dir") / "selected_cves_100.jsonl"
    docs = [build_doc(cve) for cve in tqdm(read_jsonl(source), desc="Corpus 생성")]
    write_jsonl(docs, output)
    LOG.info("%d개 문서 저장: %s", len(docs), output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build(load_config(args.config), args.force)


if __name__ == "__main__":
    main()
