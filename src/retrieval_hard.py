from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.retrieval import RetrievalEngine, result_row
from src.utils import load_config, read_jsonl, resolve_path, set_seed, setup_logging, write_jsonl

LOG = setup_logging("retrieval_hard")

# CPU 환경에서 CrossEncoder의 기본 max_length(모델 기본값, 수천 토큰까지 허용)는
# 20개 후보 재랭킹에도 질의당 수분이 걸릴 정도로 비실용적이다. bge-reranker-v2-m3는
# 입력을 뒤에서 자르므로, CVE corpus 문서가 "CVE ID/Description/CVSS..." 순서로
# 핵심 정보를 앞부분에 담고 있는 이 코퍼스에서는 256 토큰으로 잘라도 재랭킹 품질에
# 실질적 손실이 거의 없다. easy(500) 파이프라인의 config.yaml은 건드리지 않고
# hard 파이프라인에서만 이 값을 오버라이드한다.
HARD_RERANKER_MAX_LENGTH = 256


def run(config: dict[str, Any], force: bool = False) -> Path:
    output = resolve_path(config, "results_dir") / "retrieval_results_hard.jsonl"
    if output.exists() and not force:
        LOG.info("캐시 재사용: %s", output)
        return output
    corpus = read_jsonl(resolve_path(config, "corpus_dir") / "cve_corpus.jsonl")
    qa_rows = read_jsonl(resolve_path(config, "processed_dir") / "qa_ko_hard_400.jsonl")
    if not corpus or not qa_rows:
        raise ValueError("Corpus 또는 hard QA 데이터가 비어 있습니다.")
    config = copy.deepcopy(config)
    config["retrieval"]["reranker"]["max_length"] = HARD_RERANKER_MAX_LENGTH
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
    engine = RetrievalEngine(corpus, config)
    top_k = max(10, int(config["experiment"]["top_k_retrieval"]))
    rows = []
    for qa in tqdm(qa_rows, desc="Hard Retrieval", unit="query"):
        for method in methods:
            indices, scores = engine.search(qa["question_ko"], method, top_k)
            row = result_row(qa, method, indices, scores, corpus)
            row["question_type"] = qa.get("question_type")
            row["attack_profile_ambiguity_group_size"] = qa.get("attack_profile_ambiguity_group_size")
            rows.append(row)
    write_jsonl(rows, output)
    LOG.info("%d개 hard 검색 결과 저장: %s", len(rows), output)
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
