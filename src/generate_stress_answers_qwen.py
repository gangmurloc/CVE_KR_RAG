from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.generate_answers_qwen import (
    FIXED_MODEL, SYSTEM_PROMPT, generate_one, load_model, make_result, user_prompt,
)
from src.utils import (
    load_config, read_jsonl, resolve_path, set_seed, setup_logging, write_jsonl,
)

LOG = setup_logging("generate_stress_answers_qwen")


def annotate(row: dict[str, Any], qa: dict[str, Any]) -> dict[str, Any]:
    row["stress_type"] = qa["stress_type"]
    row["expected_answerable"] = bool(qa["expected_answerable"])
    return row


def run(config: dict[str, Any], force: bool = False) -> Path:
    results = resolve_path(config, "results_dir")
    output = results / "stress_generated_answers_qwen.jsonl"
    qa_rows = read_jsonl(
        resolve_path(config, "processed_dir") / "qa_ko_stress_200.jsonl"
    )
    qa_by_id = {row["qa_id"]: row for row in qa_rows}
    docs = read_jsonl(resolve_path(config, "corpus_dir") / "cve_corpus.jsonl")
    docs_by_id = {doc["doc_id"]: doc for doc in docs}
    retrieval = read_jsonl(results / "stress_retrieval_results.jsonl")
    existing = [] if force or not output.exists() else read_jsonl(output)
    existing = [row for row in existing if not row.get("generation_error")]
    completed = {(row["qa_id"], row["method"]) for row in existing}
    tasks = [
        row for row in retrieval
        if (row["qa_id"], row["method"]) not in completed
    ]
    if not tasks:
        LOG.info("스트레스 생성 캐시가 완전합니다: %s", output)
        return output
    try:
        tokenizer, model, load_mode = load_model(config)
        LOG.info("%s 로딩 완료 (%s)", FIXED_MODEL, load_mode)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        LOG.error("Qwen 모델 로딩 실패: %s", error)
        (results / "stress_generation_error.log").write_text(
            error + "\n\n" + traceback.format_exc(), encoding="utf-8"
        )
        for result in tasks:
            qa = qa_by_id[result["qa_id"]]
            context_ids = result["retrieved_doc_ids"][
                : int(config["generation"]["context_top_k"])
            ]
            existing.append(annotate(
                make_result(qa, result["method"], context_ids, error=error), qa
            ))
        write_jsonl(existing, output)
        return output

    context_k = int(config["generation"]["context_top_k"])
    for index, result in enumerate(
        tqdm(tasks, desc="Stress Qwen 생성", unit="answer"), 1
    ):
        qa = qa_by_id[result["qa_id"]]
        context_ids = result["retrieved_doc_ids"][:context_k]
        contexts = [docs_by_id[doc_id] for doc_id in context_ids if doc_id in docs_by_id]
        try:
            raw = generate_one(
                tokenizer,
                model,
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt(qa, contexts)},
                ],
                config["generation"],
            )
            row = make_result(qa, result["method"], context_ids, raw=raw)
        except Exception as exc:
            LOG.exception(
                "스트레스 생성 실패: qa_id=%s method=%s",
                qa["qa_id"], result["method"],
            )
            row = make_result(
                qa, result["method"], context_ids,
                error=f"{type(exc).__name__}: {exc}",
            )
        existing.append(annotate(row, qa))
        if index % 10 == 0:
            write_jsonl(existing, output)
    write_jsonl(existing, output)
    LOG.info("%d개 스트레스 생성 결과 저장: %s", len(existing), output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(int(config["experiment"]["seed"]))
    run(config, args.force)


if __name__ == "__main__":
    main()
