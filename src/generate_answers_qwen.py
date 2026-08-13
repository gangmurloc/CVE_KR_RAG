from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.utils import (
    extract_json_object, load_config, read_jsonl, resolve_path, set_seed, setup_logging, write_jsonl,
)

LOG = setup_logging("generate_answers_qwen")
FIXED_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SYSTEM_PROMPT = """너는 사이버보안 취약점 분석 보조자다.
반드시 제공된 근거 문서에 기반해서만 답변하라.
근거 문서에 없는 내용은 추측하지 말고 "제공된 근거만으로는 확인할 수 없음"이라고 답하라.
답변은 반드시 한국어로 작성하라.
출력은 반드시 JSON 형식이어야 한다.
JSON 이외의 설명 문장을 출력하지 마라."""


def user_prompt(qa: dict[str, Any], contexts: list[dict[str, Any]]) -> str:
    rendered = "\n\n".join(
        f"[doc_id: {doc['doc_id']}]\n{doc['text']}" for doc in contexts
    )
    return f"""다음은 CVE 취약점 관련 검색 문서이다.

[Context]
{rendered}

[Question]
{qa['question_ko']}

다음 JSON schema를 엄격히 따라 답변하라.

{{
  "qa_id": "{qa['qa_id']}",
  "cve_id": "{qa['cve_id']}",
  "answer_ko": "한국어 답변",
  "cited_doc_ids": ["근거로 사용한 doc_id"],
  "extracted_fields": {{
    "severity": "CRITICAL/HIGH/MEDIUM/LOW 또는 확인 불가",
    "cvss_score": "숫자 또는 확인 불가",
    "attack_vector": "NETWORK/ADJACENT/LOCAL/PHYSICAL 또는 확인 불가",
    "privileges_required": "NONE/LOW/HIGH 또는 확인 불가",
    "user_interaction": "NONE/REQUIRED 또는 확인 불가",
    "cwe_ids": ["CWE-ID"]
  }},
  "uncertainty": "확실/부분 확인/확인 불가"
}}

주의:
- context에 없는 내용을 추가하지 마라.
- references만 있고 실제 내용이 없으면 그 사실을 명시하라.
- cited_doc_ids에는 반드시 context에 포함된 doc_id만 넣어라.
- JSON parsing이 가능하도록 큰따옴표를 사용하라."""


def load_model(config: dict[str, Any]) -> tuple[Any, Any, str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    generation = config["generation"]
    configured = generation.get("model_name")
    if configured != FIXED_MODEL:
        raise ValueError(f"생성 모델은 연구 설계상 {FIXED_MODEL}로 고정되어야 합니다: {configured}")
    tokenizer = AutoTokenizer.from_pretrained(FIXED_MODEL)
    common: dict[str, Any] = {
        "device_map": generation.get("device", "auto"),
        "torch_dtype": generation.get("torch_dtype", "auto"),
    }
    load_mode = "standard"
    if generation.get("load_in_4bit", True):
        try:
            from transformers import BitsAndBytesConfig

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                FIXED_MODEL, quantization_config=quant_config, **common
            )
            load_mode = "4bit"
            return tokenizer, model, load_mode
        except Exception as exc:
            LOG.warning("4-bit 모델 로딩 실패, 일반 로딩을 시도합니다: %s", exc)
    model = AutoModelForCausalLM.from_pretrained(FIXED_MODEL, **common)
    return tokenizer, model, load_mode


def model_device(model: Any) -> Any:
    try:
        return model.device
    except AttributeError:
        return next(model.parameters()).device


def generate_one(tokenizer: Any, model: Any, messages: list[dict[str, str]], cfg: dict[str, Any]) -> str:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    device = model_device(model)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    kwargs: dict[str, Any] = {
        "max_new_tokens": int(cfg["max_new_tokens"]),
        "do_sample": bool(cfg["do_sample"]),
    }
    if kwargs["do_sample"]:
        kwargs.update(temperature=float(cfg["temperature"]), top_p=float(cfg["top_p"]))
    output = model.generate(**inputs, **kwargs)
    generated = output[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def make_result(
    qa: dict[str, Any], method: str, context_ids: list[str], raw: str = "",
    error: str | None = None,
) -> dict[str, Any]:
    parsed, recovered = extract_json_object(raw) if raw else (None, False)
    return {
        "qa_id": qa["qa_id"],
        "cve_id": qa["cve_id"],
        "question_type": qa.get("question_type"),
        "method": method,
        "context_doc_ids": context_ids,
        "raw_model_output": raw,
        "parsed_json": parsed,
        "parse_success": parsed is not None,
        "json_recovered": recovered,
        "answer_ko": parsed.get("answer_ko", "") if parsed else "",
        "cited_doc_ids": parsed.get("cited_doc_ids", []) if parsed else [],
        "extracted_fields": parsed.get("extracted_fields", {}) if parsed else {},
        "generation_error": error,
    }


def run(config: dict[str, Any], force: bool = False) -> Path:
    results_dir = resolve_path(config, "results_dir")
    output = results_dir / "generated_answers_qwen.jsonl"
    qa_rows = read_jsonl(resolve_path(config, "processed_dir") / "qa_ko_500.jsonl")
    qa_by_id = {row["qa_id"]: row for row in qa_rows}
    docs = read_jsonl(resolve_path(config, "corpus_dir") / "cve_corpus.jsonl")
    docs_by_id = {doc["doc_id"]: doc for doc in docs}
    retrieval = read_jsonl(results_dir / "retrieval_results.jsonl")
    existing = [] if force or not output.exists() else read_jsonl(output)
    # Preserve expensive successful generations and parse failures with raw output,
    # but retry rows that only represent a prior loading/generation error.
    existing = [row for row in existing if not row.get("generation_error")]
    completed = {(row["qa_id"], row["method"]) for row in existing}
    tasks = [row for row in retrieval if (row["qa_id"], row["method"]) not in completed]
    if not tasks:
        LOG.info("생성 캐시가 완전합니다: %s", output)
        return output

    try:
        tokenizer, model, load_mode = load_model(config)
        LOG.info("%s 로딩 완료 (%s)", FIXED_MODEL, load_mode)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        LOG.error("Qwen 모델 로딩 실패; generation을 건너뜁니다: %s", error)
        (results_dir / "generation_error.log").write_text(
            error + "\n\n" + traceback.format_exc(), encoding="utf-8"
        )
        for result in tasks:
            qa = qa_by_id[result["qa_id"]]
            context_ids = result["retrieved_doc_ids"][: int(config["generation"]["context_top_k"])]
            existing.append(make_result(qa, result["method"], context_ids, error=error))
        write_jsonl(existing, output)
        return output

    context_k = int(config["generation"]["context_top_k"])
    for index, result in enumerate(tqdm(tasks, desc="Qwen 답변 생성", unit="answer"), 1):
        qa = qa_by_id[result["qa_id"]]
        context_ids = result["retrieved_doc_ids"][:context_k]
        contexts = [docs_by_id[doc_id] for doc_id in context_ids if doc_id in docs_by_id]
        try:
            raw = generate_one(
                tokenizer, model,
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": user_prompt(qa, contexts)}],
                config["generation"],
            )
            row = make_result(qa, result["method"], context_ids, raw=raw)
        except Exception as exc:
            LOG.exception("생성 실패: qa_id=%s method=%s", qa["qa_id"], result["method"])
            row = make_result(qa, result["method"], context_ids, error=f"{type(exc).__name__}: {exc}")
        existing.append(row)
        if index % 10 == 0:
            write_jsonl(existing, output)
    write_jsonl(existing, output)
    LOG.info("%d개 생성 결과 저장: %s", len(existing), output)
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
