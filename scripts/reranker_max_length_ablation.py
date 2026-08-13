"""Checks whether retrieval_hard.py's HARD_RERANKER_MAX_LENGTH=256 override
(needed to make CPU reranking of the 400-question hard split tractable, see
retrieval_hard.py) is a confound when comparing hybrid+reranker results
between the easy split (default, unbounded max_length) and the hard split
(256). Reranks a fixed random sample of easy-split queries with both
settings and reports whether the top-1 pick and gold rank change.

Default max_length is extremely slow on CPU for this reranker (roughly
150-250s per query for 20 candidates), so this intentionally uses a small
sample rather than the full 500-question set.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np
from sentence_transformers import CrossEncoder

from src.retrieval import RetrievalEngine
from src.utils import load_config, read_jsonl, resolve_path, set_seed, setup_logging

LOG = setup_logging("reranker_max_length_ablation")


def run(config: dict[str, Any], n_sample: int, seed: int) -> None:
    corpus = read_jsonl(resolve_path(config, "corpus_dir") / "cve_corpus.jsonl")
    qa_rows = read_jsonl(resolve_path(config, "processed_dir") / "qa_ko_500.jsonl")

    rng = np.random.RandomState(seed)
    sample = [qa_rows[i] for i in rng.choice(len(qa_rows), size=n_sample, replace=False)]

    engine = RetrievalEngine(corpus, config)
    top_n = int(config["retrieval"]["reranker"]["top_n_before_rerank"])
    model_name = config["retrieval"]["reranker"]["model_name"]

    candidates = [(qa, np.argsort(-engine.hybrid_scores(qa["question_ko"]), kind="stable")[:top_n]) for qa in sample]

    results = {}
    for max_length, label in [(None, "default"), (256, "256")]:
        kwargs = {"max_length": max_length} if max_length else {}
        reranker = CrossEncoder(model_name, **kwargs)
        per_query, t0 = [], time.time()
        for qa, order in candidates:
            pairs = [(qa["question_ko"], corpus[int(i)]["text"]) for i in order]
            scores = np.asarray(reranker.predict(pairs, show_progress_bar=False), dtype=np.float64)
            rerank_order = order[np.argsort(-scores, kind="stable")]
            cve_ids = [corpus[int(i)]["cve_id"] for i in rerank_order]
            rank = next((i for i, v in enumerate(cve_ids, 1) if v == qa["cve_id"]), None)
            per_query.append({"qa_id": qa["qa_id"], "top1": cve_ids[0], "rank": rank})
        elapsed = time.time() - t0
        results[label] = per_query
        LOG.info("max_length=%s: %.1fs for %d queries", label, elapsed, n_sample)
        del reranker

    agree = sum(a["top1"] == b["top1"] for a, b in zip(results["default"], results["256"]))
    LOG.info("top1 agreement between default and 256: %d/%d", agree, n_sample)
    for a, b in zip(results["default"], results["256"]):
        LOG.info("%s: default_rank=%s 256_rank=%s", a["qa_id"], a["rank"], b["rank"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--n-sample", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(args.seed)
    run(config, args.n_sample, args.seed)


if __name__ == "__main__":
    main()
