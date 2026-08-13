from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from src.retrieval import RetrievalEngine, result_row
from src.utils import (
    load_config, read_jsonl, resolve_path, set_seed, setup_logging, write_jsonl,
)

LOG = setup_logging("retrieval_stress")


def enabled_methods(config: dict[str, Any]) -> list[str]:
    cfg = config["retrieval"]
    methods = []
    if cfg["bm25"].get("enabled"):
        methods.append("bm25")
    if cfg["dense"].get("enabled"):
        methods.append("dense")
    if cfg["hybrid"].get("enabled"):
        methods.append("hybrid")
    if cfg["reranker"].get("enabled"):
        methods.append("hybrid_reranker")
    return methods


def original_qa_id(qa: dict[str, Any]) -> str:
    index_by_type = {
        "vulnerability_summary": 1,
        "affected_product": 2,
        "attack_condition": 3,
        "severity_reason": 4,
        "mitigation": 5,
    }
    return f"{qa['cve_id']}-Q{index_by_type[qa['question_type']]}"


def removed_evidence_row(
    qa: dict[str, Any],
    method: str,
    original: dict[str, Any],
    docs_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    kept = [
        (doc_id, cve_id, score)
        for doc_id, cve_id, score in zip(
            original["retrieved_doc_ids"],
            original["retrieved_cve_ids"],
            original["scores"],
        )
        if cve_id != qa["cve_id"]
    ][:10]
    docs = [docs_by_id[doc_id] for doc_id, _, _ in kept]
    # result_row only requires indices into the supplied document list.
    row = result_row(
        qa,
        method,
        indices=np.arange(len(docs)),
        scores=np.asarray([score for _, _, score in kept]),
        docs=docs,
    )
    return row


def annotate(row: dict[str, Any], qa: dict[str, Any]) -> dict[str, Any]:
    row["stress_type"] = qa["stress_type"]
    row["expected_answerable"] = bool(qa["expected_answerable"])
    return row


def run(config: dict[str, Any], force: bool = False) -> Path:
    results = resolve_path(config, "results_dir")
    output = results / "stress_retrieval_results.jsonl"
    existing = [] if force or not output.exists() else read_jsonl(output)
    completed = {(row["qa_id"], row["method"]) for row in existing}
    stress_qa = read_jsonl(
        resolve_path(config, "processed_dir") / "qa_ko_stress_200.jsonl"
    )
    docs = read_jsonl(resolve_path(config, "corpus_dir") / "cve_corpus.jsonl")
    docs_by_id = {doc["doc_id"]: doc for doc in docs}
    methods = enabled_methods(config)
    original = {
        (row["qa_id"], row["method"]): row
        for row in read_jsonl(results / "retrieval_results.jsonl")
    }
    tasks = [
        (qa, method)
        for qa in stress_qa
        for method in methods
        if (qa["qa_id"], method) not in completed
    ]
    if not tasks:
        LOG.info("스트레스 검색 캐시가 완전합니다: %s", output)
        return output
    engine = RetrievalEngine(docs, config)
    top_k = max(10, int(config["experiment"]["top_k_retrieval"]))
    for task_index, (qa, method) in enumerate(
        tqdm(tasks, desc="Stress Retrieval", unit="query-method"), 1
    ):
        if qa["stress_type"] == "gold_evidence_removed":
            key = (original_qa_id(qa), method)
            if key not in original:
                raise KeyError(f"원본 검색 결과가 없습니다: {key}")
            row = removed_evidence_row(qa, method, original[key], docs_by_id)
        else:
            indices, scores = engine.search(qa["question_ko"], method, top_k)
            row = result_row(qa, method, indices, scores, docs)
        existing.append(annotate(row, qa))
        if task_index % 20 == 0:
            write_jsonl(existing, output)
    write_jsonl(existing, output)
    LOG.info("%d개 스트레스 검색 결과 저장: %s", len(existing), output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    config = load_config(args.config)
    set_seed(int(config["experiment"]["seed"]))
    run(config, args.force)


if __name__ == "__main__":
    main()
