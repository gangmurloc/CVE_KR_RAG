from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from src.utils import load_config, minmax, read_jsonl, resolve_path, set_seed, setup_logging, write_jsonl

LOG = setup_logging("retrieval")
CVE_PATTERN = re.compile(r"CVE-\d{4}-\d{4,}", re.IGNORECASE)


def tokenize_kr_direct(text: str) -> list[str]:
    normalized = text.upper()
    cve_tokens = CVE_PATTERN.findall(normalized)
    words = re.findall(r"[A-Z0-9_.:/-]+|[가-힣]+", normalized)
    korean_ngrams: list[str] = []
    for word in re.findall(r"[가-힣]+", normalized):
        korean_ngrams.extend(word[i : i + 2] for i in range(max(0, len(word) - 1)))
    # Repeat explicit CVE IDs to ensure exact identifiers strongly affect BM25.
    return words + korean_ngrams + cve_tokens * 2


class RetrievalEngine:
    def __init__(self, docs: list[dict[str, Any]], config: dict[str, Any]) -> None:
        self.docs = docs
        self.texts = [doc["text"] for doc in docs]
        self.config = config
        self._bm25: Any = None
        self._dense_model: Any = None
        self._doc_embeddings: np.ndarray | None = None
        self._reranker: Any = None
        self._reranker_failed = False

    def prepare_bm25(self) -> None:
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi([tokenize_kr_direct(text) for text in self.texts])

    def bm25_scores(self, query: str) -> np.ndarray:
        self.prepare_bm25()
        return np.asarray(self._bm25.get_scores(tokenize_kr_direct(query)), dtype=np.float64)

    def prepare_dense(self) -> None:
        if self._dense_model is not None:
            return
        from sentence_transformers import SentenceTransformer

        cfg = self.config["retrieval"]["dense"]
        model_name = cfg["embedding_model"]
        self._dense_model = SentenceTransformer(model_name)
        corpus_dir = resolve_path(self.config, "corpus_dir")
        fingerprint = hashlib.sha256(
            (model_name + "\n" + "\n".join(doc["doc_id"] for doc in self.docs)).encode()
        ).hexdigest()[:16]
        cache = corpus_dir / f"dense_embeddings_{fingerprint}.npy"
        if cache.exists():
            self._doc_embeddings = np.load(cache)
            if self._doc_embeddings.shape[0] == len(self.docs):
                LOG.info("Dense embedding 캐시 재사용: %s", cache)
                return
        self._doc_embeddings = self._dense_model.encode(
            self.texts,
            batch_size=int(cfg["batch_size"]),
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        np.save(cache, self._doc_embeddings)

    def dense_scores(self, query: str) -> np.ndarray:
        self.prepare_dense()
        assert self._dense_model is not None and self._doc_embeddings is not None
        query_embedding = self._dense_model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )[0]
        return np.asarray(self._doc_embeddings @ query_embedding, dtype=np.float64)

    def hybrid_scores(self, query: str) -> np.ndarray:
        alpha = float(self.config["retrieval"]["hybrid"]["alpha"])
        return alpha * minmax(self.bm25_scores(query)) + (1.0 - alpha) * minmax(self.dense_scores(query))

    def rerank(self, query: str, indices: np.ndarray, fallback_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._reranker_failed:
            return indices, fallback_scores[indices]
        try:
            if self._reranker is None:
                from sentence_transformers import CrossEncoder

                reranker_cfg = self.config["retrieval"]["reranker"]
                max_length = reranker_cfg.get("max_length")
                kwargs = {"max_length": max_length} if max_length else {}
                self._reranker = CrossEncoder(reranker_cfg["model_name"], **kwargs)
            pairs = [(query, self.texts[int(index)]) for index in indices]
            scores = np.asarray(self._reranker.predict(pairs, show_progress_bar=False), dtype=np.float64)
            order = np.argsort(-scores, kind="stable")
            return indices[order], scores[order]
        except Exception as exc:
            self._reranker_failed = True
            LOG.warning("Reranker 사용 실패; Hybrid 순위를 유지합니다: %s", exc)
            return indices, fallback_scores[indices]

    def search(self, query: str, method: str, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if method == "bm25":
            scores = self.bm25_scores(query)
            order = np.argsort(-scores, kind="stable")
            return order[:top_k], scores[order[:top_k]]
        if method == "dense":
            scores = self.dense_scores(query)
            order = np.argsort(-scores, kind="stable")
            return order[:top_k], scores[order[:top_k]]
        scores = self.hybrid_scores(query)
        order = np.argsort(-scores, kind="stable")
        if method == "hybrid":
            return order[:top_k], scores[order[:top_k]]
        if method == "hybrid_reranker":
            top_n = int(self.config["retrieval"]["reranker"]["top_n_before_rerank"])
            reranked, rerank_scores = self.rerank(query, order[:top_n], scores)
            return reranked[:top_k], rerank_scores[:top_k]
        raise ValueError(f"알 수 없는 retrieval method: {method}")


def result_row(
    qa: dict[str, Any], method: str, indices: np.ndarray, scores: np.ndarray,
    docs: list[dict[str, Any]],
) -> dict[str, Any]:
    retrieved = [docs[int(index)] for index in indices]
    cve_ids = [doc["cve_id"] for doc in retrieved]
    gold = qa["cve_id"]
    rank = next((i for i, value in enumerate(cve_ids, 1) if value == gold), None)
    return {
        "qa_id": qa["qa_id"],
        "cve_id": gold,
        "question_type": qa.get("question_type"),
        "method": method,
        "query": qa["question_ko"],
        "retrieved_doc_ids": [doc["doc_id"] for doc in retrieved],
        "retrieved_cve_ids": cve_ids,
        "scores": [float(score) for score in scores],
        "retrieved_doc_ids_top5": [doc["doc_id"] for doc in retrieved[:5]],
        "retrieved_cve_ids_top5": cve_ids[:5],
        "scores_top5": [float(score) for score in scores[:5]],
        "retrieved_doc_ids_top10": [doc["doc_id"] for doc in retrieved[:10]],
        "retrieved_cve_ids_top10": cve_ids[:10],
        "scores_top10": [float(score) for score in scores[:10]],
        "top1_cve_id": cve_ids[0] if cve_ids else None,
        "hit_at_1": bool(rank and rank <= 1),
        "hit_at_5": bool(rank and rank <= 5),
        "hit_at_10": bool(rank and rank <= 10),
        "rank_of_gold_cve": rank,
    }


def run(config: dict[str, Any], force: bool = False) -> Path:
    output = resolve_path(config, "results_dir") / "retrieval_results.jsonl"
    if output.exists() and not force:
        LOG.info("캐시 재사용: %s", output)
        return output
    corpus = read_jsonl(resolve_path(config, "corpus_dir") / "cve_corpus.jsonl")
    qa_rows = read_jsonl(resolve_path(config, "processed_dir") / "qa_ko_500.jsonl")
    if not corpus or not qa_rows:
        raise ValueError("Corpus 또는 QA 데이터가 비어 있습니다.")
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
    for qa in tqdm(qa_rows, desc="Retrieval", unit="query"):
        for method in methods:
            indices, scores = engine.search(qa["question_ko"], method, top_k)
            rows.append(result_row(qa, method, indices, scores, corpus))
    write_jsonl(rows, output)
    LOG.info("%d개 검색 결과 저장: %s", len(rows), output)
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
