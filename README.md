# 한국어 CVE RAG 검색 전략별 신뢰성 평가

## Abstract (English)

A reproducible evaluation pipeline for Korean-language CVE question
answering under four retrieval strategies — BM25, dense retrieval
(BAAI/bge-m3), hybrid (BM25+dense), and hybrid with a cross-encoder
reranker (BAAI/bge-reranker-v2-m3) — measuring both retrieval quality and
downstream answer reliability with a fixed generation model
(Qwen/Qwen2.5-7B-Instruct). Data is collected live from the NVD CVE API and
the CISA Known Exploited Vulnerabilities catalog. All reported metrics are
generated from the released evaluation pipeline and corresponding result
artifacts.

**Main finding.** On the original 500-question benchmark, BM25 reaches
Hit@1 = 1.000, which looks like an outright win — but every question embeds
the gold CVE ID verbatim, and the BM25 tokenizer explicitly up-weights CVE
ID tokens, so this measures lexical exact-match on a unique identifier, not
retrieval skill. To isolate the real effect, this repo adds a second,
400-question **hard split** that never mentions the CVE ID and instead
describes each vulnerability only by product/vendor/version, CVSS attack
conditions (attack vector, privileges required, user interaction), and
severity/impact. Once the identifier is removed, BM25's Hit@1 drops to
0.67–0.73 (depending on evaluation weighting, see below) and BM25 is no
longer the top method on any metric — a different ranking from the easy
split, where BM25 dominates across the board.

Some hard-split templates strip enough information that more than one CVE
in the 100-document corpus can produce byte-identical question text (most
acutely for `attribute_only_hard`, which drops the product name entirely —
34 of 100 CVEs share one attack-vector/privileges/user-interaction/impact
profile; 115 of 400 hard questions have more than one gold CVE overall).
Retrieval is scored against the **set** of all CVEs whose deterministic
template output matches the question verbatim (`gold_cve_id_set`), not the
single CVE that happened to generate it — otherwise a semantically correct
retrieval of an indistinguishable sibling CVE would be counted as a miss.
This also means the standard single-gold Recall@k/nDCG@k formulas do not
apply as-is; `metrics_for_multigold()` recomputes them as true set-recall
and multi-relevant nDCG, and `Hit@k` is reported alongside `Recall@k` as a
distinct, non-equivalent metric once a query can have multiple correct
answers.

Because several rows share an identical (question_type, question_text) pair
(e.g. the 34-CVE `attribute_only_hard` group), simply averaging over all 400
generated rows (**CVE-weighted**) lets one shared question outweigh a unique
one 34-to-1. Collapsing duplicates down to the 307 distinct queries
(**query-weighted**) changes which method looks best: CVE-weighted ranks
Hybrid highest on Hit@1 (0.7050) and Hybrid+Reranker lowest (0.6500);
query-weighted — arguably the more defensible unit of evaluation — ranks
**Hybrid+Reranker highest on most metrics** (Hit@1 = 0.7883), while tying
Hybrid on Hit@10 and trailing it slightly on Recall@10. Both weightings
agree on the headline diagnosis (BM25 loses its lead once the ID is
removed); they disagree on which retrieval strategy is actually best, which
is itself evidence that evaluation-unit choice matters as much as the
identifier-shortcut fix.

| Split | Weighting | Method | Hit@1 | MRR@10 |
| --- | --- | --- | --- | --- |
| Easy (500, CVE ID in question) | — | BM25 | 1.0000 | 1.0000 |
| Easy (500, CVE ID in question) | — | Hybrid + Reranker | 0.9940 | 0.9967 |
| Hard (400, no CVE ID) | CVE-weighted | Hybrid | 0.7050 | 0.7447 |
| Hard (400, no CVE ID) | Query-weighted (307 unique) | Hybrid + Reranker | 0.7883 | 0.8272 |

Full tables (retrieval, answer accuracy, evidence-gated abstention,
per-template and per-weighting hard-split breakdowns) are under
`data/results/`. Setup instructions are in `REPRODUCIBILITY.md`. The rest of
this README, including the full Method/Limitations sections, is in Korean.

---

## 연구 목적

한국어 CVE 보안 질의에 대해 BM25, Dense Retrieval, Hybrid Retrieval,
Hybrid+Reranker가 검색 성능과 최종 답변 신뢰성에 미치는 영향을 동일한
데이터·프롬프트·생성 설정 아래 평가하는 재현 가능한 Python 실험 파이프라인이다.

“본 연구는 생성 모델 성능 비교가 아니라 RAG 검색 전략의 영향을 분석하는 것을 목적으로 하므로, 생성 LLM은 Qwen/Qwen2.5-7B-Instruct로 고정하였다.”

실험은 새 모델을 학습하지 않는다. NVD와 CISA KEV에서 실제 데이터를 수집하며,
코드는 측정되지 않은 수치나 가짜 생성 결과를 만들지 않는다.

## Key Results

100개 CVE × 5개 고정 템플릿(총 500 QA) 기준. 전체 표는 `data/results/`에 있다.

| Method | Retrieval Hit@1 | Retrieval MRR@10 | Answer Field Accuracy | Citation Accuracy |
| --- | --- | --- | --- | --- |
| BM25 | 1.0000 | 1.0000 | 0.9120 | 0.9220 |
| Dense (BGE-M3) | 0.4000 | 0.4987 | 0.6217 | 0.6120 |
| Hybrid | 0.9980 | 0.9990 | 0.9363 | 0.9420 |
| Hybrid + Reranker | 0.9940 | 0.9967 | 0.9473 | 0.9560 |

이 500개 질문은 모두 `{cve_id}`를 질문 문자열에 직접 포함하고, BM25 토크나이저가
CVE ID 토큰을 명시적으로 가중(`retrieval.py`의 `tokenize_kr_direct`)하기 때문에
BM25의 Hit@1 = 1.000은 검색 전략의 우수성이 아니라 **질문에 포함된 고유 식별자를
그대로 되찾는 lexical exact-match**의 결과다. Dense retrieval만 놓고 보면
오히려 BM25보다 낮은 성능(Hit@1 0.400)을 보이는데, 이는 식별자 shortcut이 없을 때
검색 전략들이 실제로 어떻게 갈리는지가 이 500개 질문만으로는 보이지 않는다는
뜻이기도 하다.

### Hard split: CVE ID를 제거한 속성 기반 질의

위 문제를 진단하기 위해 `{cve_id}`를 전혀 포함하지 않는 400개의 질문
(`data/processed/qa_ko_hard_400.csv`, CVE당 4개 템플릿)을 추가했다. 제품/벤더/버전,
CVSS 공격 조건(AV/PR/UI), 심각도·영향도만으로 문서를 찾아야 하며, 템플릿별로
식별 가능한 정보량을 단계적으로 줄인다(`src/build_qa_dataset_hard.py`).

| 난이도 | 질문 예시 | 포함 정보 |
| --- | --- | --- |
| product_version_hard | "Oracle Coherence (3.7.1.0 버전)에 존재하는 보안 취약점은 무엇인가?" | 벤더+제품+버전 |
| severity_scenario_hard | "...CVSS 기본점수 9.8(CRITICAL 등급)의 취약점으로..." | 벤더+제품+CVSS |
| attack_condition_hard | "...네트워크를 통해 원격으로, 인증 없이, 사용자 상호작용 없이 악용될 수 있는..." | 벤더+제품+공격 조건 |
| attribute_only_hard | "네트워크를 통해 원격으로, 인증 없이... 악용 가능하며 기밀성·무결성·가용성에 심각한 영향을 주는 취약점은?" | 공격 조건만(제품명 없음) |

### 평가 방법: gold set 기반 채점 (단일 CVE 채점의 함정)

`attribute_only_hard`는 벤더/제품명을 전혀 포함하지 않으므로, 동일한 공격
조건·영향도 조합을 가진 CVE가 여러 개면 그 문항은 원칙적으로 정답이 여러 개다
(product_version_hard 등 나머지 세 템플릿도 드물게 같은 문제가 생긴다 — 예를
들어 같은 벤더/제품/버전을 공유하는 CVE 두 개가 있으면 동일한 질문 텍스트가
나온다). 이런 상태에서 "이 문항을 생성한 CVE 한 개"만 정답으로 놓고 Hit@1을
계산하면, 검색기가 의미상 완전히 맞는(동일 텍스트를 만들어내는) 다른 CVE를
1위로 올려도 오답으로 처리되어 지표가 실제 성능보다 낮게 나온다.

이를 바로잡기 위해 `src/build_qa_dataset_hard.py`는 QA를 모두 생성한 뒤
(question_type, question_ko) 완전히 동일한 텍스트를 만들어내는 모든 CVE를
모아 `gold_cve_id_set`으로 저장한다. `src/retrieval_hard.py`의
`result_row_hard()`는 검색 결과의 상위 문서 중 이 집합에 속하는 첫 번째
순위를 `rank_of_gold_cve`로 사용한다. 400개 hard 질문 중 115개(`gold_set_size > 1`)가
실제로 둘 이상의 정답을 가지며, 그중 34개는 `attribute_only_hard`에서 하나의
공격 조건·영향도 조합을 공유하는 최대 그룹이다.

gold set이 여러 개일 수 있다는 사실은 지표 정의에도 영향을 준다.

- **Hit@k**(=top-k 안에 gold set 중 하나라도 있으면 1)와 **MRR@10**(첫 번째로
  맞힌 순위의 역수)은 gold가 여러 개여도 정의가 그대로 유효한 표준 IR 지표다.
- 반면 **Recall@k**는 "gold set 중 top-k에 들어온 것의 비율"
  (`|top-k ∩ gold| / |gold|`)이어야 한다. gold가 하나뿐이면 이 값과 Hit@k가
  같아지지만, gold가 여러 개면 서로 다른 지표다 — 예를 들어 gold 4개 중 top-5에
  1개만 들어오면 Recall@5 = 0.25이지 1.0이 아니다. **nDCG@10**도 같은 이유로
  top-10 안에 있는 gold 문서 전부의 위치를 반영한
  `DCG@10 = Σ 1/log2(rank+1)` (해당 rank의 문서가 gold일 때만) 과
  `IDCG@10 = Σ_{i=1..min(|gold|,10)} 1/log2(i+1)`의 비율로 계산해야 한다.
  `evaluate_retrieval_hard.py`의 `metrics_for_multigold()`가 이 두 지표를
  올바르게 구현하고, `Hit@5`/`Hit@10`은 기존 "Recall"이라는 이름이 실제로
  의미하던 것(=any-hit)을 그대로 별도 컬럼으로 남긴다.

또한 400개 행 중 다수는 같은 (question_type, question_ko)를 gold set 크기만큼
반복해서 담고 있다 — 예를 들어 위에서 언급한 34-CVE 그룹은 완전히 동일한 질의·
gold set을 가진 행이 34번 들어 있다. CVE 1개당 질문 1개라는 원 설계를 그대로
평균 내면(**CVE-weighted**, 400행), 이 하나의 attribute_only_hard 그룹이 다른
독립적인 질의 33개와 같은 가중치를 갖게 된다. 이는 임의로 정의할 수 있는 평가
단위이긴 하지만, gold를 "질문이 실제로 구별할 수 있는 정답 집합"으로 재정의한
이상 "구별 가능한 질의 하나"를 한 단위로 보는 평가(**query-weighted**,
(question_type, question_ko) 기준 dedup, 400개 중 307개 unique)가 더 자연스럽다.
아래에는 두 결과를 모두 낸다.

**Hard split 결과 — CVE-weighted (400 질의, 중복 포함):**

| Method | Hit@1 | Hit@5 | Hit@10 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BM25 | 0.6700 | 0.7475 | 0.7650 | 0.6675 | 0.6975 | 0.7019 | 0.7058 |
| Dense (BGE-M3) | 0.6975 | 0.7775 | 0.7925 | 0.6775 | 0.7025 | 0.7305 | 0.6943 |
| Hybrid | 0.7050 | 0.8100 | 0.8325 | 0.7125 | 0.7425 | 0.7447 | 0.7177 |
| Hybrid + Reranker | 0.6500 | 0.8225 | 0.8600 | 0.7150 | 0.7425 | 0.7332 | 0.7340 |

**Hard split 결과 — query-weighted (307 unique 질의):**

| Method | Hit@1 | Hit@5 | Hit@10 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BM25 | 0.7329 | 0.8274 | 0.8502 | 0.8223 | 0.8456 | 0.7722 | 0.7899 |
| Dense (BGE-M3) | 0.7362 | 0.8306 | 0.8502 | 0.8210 | 0.8423 | 0.7769 | 0.7903 |
| Hybrid | 0.7590 | 0.8730 | 0.9023 | 0.8639 | 0.8945 | 0.8041 | 0.8245 |
| Hybrid + Reranker | **0.7883** | 0.8795 | 0.9023 | 0.8694 | 0.8880 | 0.8272 | 0.8394 |

두 표는 "어느 방법이 최고인가"라는 결론 자체를 바꾼다. CVE-weighted 기준으로는
Hybrid(0.7050)가 Hit@1 1위이고 Hybrid+Reranker(0.6500)가 오히려 가장 낮았지만,
중복 질의를 collapse한 query-weighted 기준으로는 **Hybrid+Reranker(0.7883)가
대부분의 지표에서 가장 높고**, Hit@10은 Hybrid와 동률이며 Recall@10은 Hybrid가
약간 더 높다. 이 반전은 CVE-weighted 평균에서 `attribute_only_hard`의 34-CVE
그룹(반복 34회)이 Hybrid+Reranker에 특히 불리하게 반복 집계됐기 때문이며,
독립적인 307개 질의로 보면 Hybrid+Reranker가 대체로 가장 강하다는 것이 더
신뢰할 수 있는 결론이다. 두 관점 모두 공통적으로 확인하는 것은: CVE ID를 제거하면 BM25가
더 이상 1위가 아니라는 진단 자체는 CVE-weighted든 query-weighted든 동일하게
성립한다.

템플릿(난이도)별로 보면 그 원인이 더 분명해진다 (query-weighted, 값은 Hit@1;
전체 표는 `data/results/retrieval_metrics_hard_by_type_query_weighted.md`):

| Method | product_version (88) | severity_scenario (98) | attack_condition (92) | attribute_only (29) |
| --- | --- | --- | --- | --- |
| BM25 | 0.8295 | 0.7959 | 0.7935 | 0.0345 |
| Dense | 0.9091 | 0.7245 | 0.7935 | 0.0690 |
| Hybrid | 0.8636 | 0.8163 | 0.8152 | 0.0690 |
| Hybrid + Reranker | 0.9545 | 0.7449 | 0.9130 | 0.0345 |

(괄호는 unique 질의 수) 벤더/제품명이 남아 있는 세 템플릿에서는 모든 방법이
0.72~0.95 수준을 유지하지만, 제품명을 완전히 제거한 `attribute_only_hard`의
독립적인 29개 프로필에서는 **네 방법 모두 3~7% 수준으로 비슷하게 낮다**. 이전
CVE-weighted 표에서는 Hybrid+Reranker만 유독 낮게(0.0700) 보이고 나머지
세 방법은 0.34~0.41로 훨씬 높아 보였는데, 그 차이는 reranker의 특성이 아니라
34-CVE 그룹 하나의 (우연히 맞힌) 성공이 CVE-weighted 평균에서 34번 반복
집계됐기 때문이었다 — query-weighted로 보면 그 우위는 사라지고 네 방법이
모두 비슷하게 어려움을 겪는다. 즉 "제품명 없이는 검색 전략을 아무리 바꿔도
정답 집합 중 하나를 1위로 정확히 찍기가 원래도 어렵다"는 retrieval 자체의
한계이며, 특정 방법의 결함으로 보기는 어렵다.

### Reranker max_length: easy/hard 설정 차이가 결과를 오염시키지 않는지 확인

`src/retrieval_hard.py`는 CPU에서 20개 후보를 재랭킹하는 데 질의당 수분이
걸리는 것을 막기 위해 reranker의 `max_length`를 256으로 제한한다(easy(500)
파이프라인의 `config.yaml`은 그대로 두고 hard 파이프라인에서만 override).
이 값이 서로 다르면 easy와 hard의 Hybrid+Reranker 결과를 직접 비교할 때
"CVE ID를 뺀 효과"와 "reranker 입력 길이를 줄인 효과"가 섞일 수 있다.

이를 확인하기 위해 easy 질문 500개 중 8개를 무작위로 뽑아(`seed=42`) 동일한
hybrid top-20 후보를 default(모델 기본값) max_length와 256 각각으로 재랭킹해
비교했다(`scripts/reranker_max_length_ablation.py`). Default 설정은 8개
질의에 1325.7초(질의당 ~166초)가 걸렸고 256 설정은 35.8초(질의당 ~4.5초)가
걸렸지만, **1위 문서와 gold rank는 8개 질의 전부 두 설정에서 완전히
일치했다.** easy 질문은 CVE ID가 코퍼스 문서 텍스트의 맨 앞("CVE ID: ...")에
있어 256 토큰 이내에서도 잘리지 않으므로, 이 표본에서는 truncation이
reranker의 판단에 영향을 주지 않았다.

다만 이 확인은 표본 크기(8개)가 작고 easy 질문에서만 수행했다는 한계가
있다 — default 설정 자체가 비현실적으로 느려서(500개 전체에 적용하면
수십 시간) 전체 재현은 하지 않았다. 그래도 이 표본에서 두 설정이 완전히
일치했다는 것은, 적어도 이 코퍼스처럼 핵심 식별 정보가 문서 앞부분에 오는
구조에서는 truncation이 reranker 순위에 실질적 영향을 주지 않는다는 근거가
된다.

재현:

```bash
python scripts/run_all.py --mode hard_retrieval_only
python scripts/reranker_max_length_ablation.py
```

## 프로젝트 구조

```text
CVE_KR_RAG/
├── config.yaml
├── requirements.txt
├── .env.example
├── data/{raw,processed,corpus,results}/
├── src/
│   ├── fetch_nvd.py
│   ├── fetch_cisa_kev.py
│   ├── select_cves.py
│   ├── build_qa_dataset.py
│   ├── build_qa_dataset_hard.py
│   ├── build_stress_dataset.py
│   ├── build_corpus.py
│   ├── retrieval.py
│   ├── retrieval_hard.py
│   ├── retrieval_stress.py
│   ├── evaluate_retrieval.py
│   ├── evaluate_retrieval_hard.py
│   ├── generate_answers_qwen.py
│   ├── generate_stress_answers_qwen.py
│   ├── evaluate_answers.py
│   ├── evaluate_evidence_gate.py
│   ├── make_tables.py
│   └── utils.py
└── scripts/
    ├── run_all.py
    └── run_reliability_stress.py
```

## 설치

Python 3.10 이상을 권장한다.

```bash
git clone https://github.com/gangmurloc/CVE_KR_RAG.git
cd CVE_KR_RAG
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

4-bit Qwen 실행이 필요하면 별도로 `pip install bitsandbytes`를 실행한다.
설치가 실패해도 retrieval-only 모드는 영향을 받지 않는다. `evaluate`는 선택
의존성이며 현재 핵심 지표에는 필요하지 않다. Markdown 표를 위해 `tabulate`를
설치하면 pandas 표 렌더러를 사용하고, 없으면 내장 fallback을 사용한다.

`.env.example`을 `.env`로 복사한다.

```bash
cp .env.example .env
```

`NVD_API_KEY`는 선택 사항이다. 키가 없으면 파이프라인이 NVD 공개 제한에 맞춰
요청 간격을 늘린다. Hugging Face 접근에 토큰이 필요한 환경에서는 `HF_TOKEN`을
설정한다. API 키나 토큰을 결과 파일에 기록하지 않는다.

설치와 공개 저장소 구성이 정상인지 빠르게 확인한다.

```bash
python scripts/doctor.py
```

## 데이터 수집과 실행

기본 설정은 전체 NVD 레코드를 페이지 단위로 수집하므로 최초 실행 시간이 길 수
있다. 연구 기간을 고정하려면 `config.yaml`의 `fetch.nvd.start_date`와
`end_date`에 NVD API가 받는 ISO-8601 값을 넣는다. 원본·중간 파일은 캐시되며,
다시 실행할 때 재사용된다. 다시 수집/계산하려면 `--force`를 지정한다.

Qwen 없이 검색 평가와 논문 표를 만든다.

```bash
python scripts/run_all.py --mode retrieval_only
```

데이터 수집부터 Qwen 생성 및 답변 평가까지 실행한다.

```bash
python scripts/run_all.py --mode full_qwen
```

이미 검색 결과가 있을 때 생성 이후 단계만 실행한다.

```bash
python scripts/run_all.py --mode generation_only
```

개별 단계도 프로젝트 루트에서 `python -m src.fetch_nvd`처럼 실행할 수 있다. 각 실패는 단계 이름과
원인을 출력한다.

## GPU, VRAM 및 4-bit 양자화

BM25와 Dense Retrieval은 CPU에서도 동작한다. `BAAI/bge-m3` 및 reranker의 CPU
실행은 느릴 수 있다. Reranker 로딩이 불가능하면 경고 후 Hybrid 순위를 그대로
사용하며 그 사실을 로그에 남긴다.

Qwen2.5-7B-Instruct 일반 정밀도 실행은 상당한 RAM/VRAM이 필요하다. 기본값은
`load_in_4bit: true`이고 bitsandbytes NF4를 우선 시도한다. 4-bit 로딩 실패 시
일반 로딩을 시도하며, 그것도 실패하면 `generation_error.log`와 각 결과 행의
`generation_error`에 실제 오류를 기록하고 생성을 건너뛴다. GPU가 없거나 메모리가
부족한 시스템에서는 `retrieval_only`를 사용한다.

## 결과 파일

GitHub 저장소에는 용량이 큰 NVD/KEV 원본, 재생성 가능한 dense embedding 캐시,
전체 생성 답변, 수기 평가 작업 파일과 로그를 기본적으로 포함하지 않는다.
`data/raw/*_metadata.json`에는 수집 시점과 건수를 남기며, 원본은
`retrieval_only` 실행 시 공식 API에서 다시 수집된다. 선정 CVE·QA·코퍼스와
논문용 집계 표는 재현성 확인을 위해 저장소에 포함할 수 있다.

- `data/raw/nvd_cves.jsonl`, `cisa_kev.jsonl`: 수집 원본의 정규화본
- `data/processed/selected_cves_100.*`: 심각도·KEV·CWE 다양성 기준 선정본
- `data/processed/qa_ko_500.*`: CVE당 5개 고정 템플릿 한국어 QA (질문에 CVE ID 포함)
- `data/processed/qa_ko_hard_400.*`: CVE당 4개 속성 기반 템플릿 한국어 QA (질문에
  CVE ID 미포함). `gold_cve_id_set`/`gold_set_size` 필드에 동일 텍스트를 만드는
  모든 CVE를 기록한다(115개 질문은 정답이 2개 이상).
- `data/corpus/cve_corpus.jsonl`: 검색 문서 및 메타데이터
- `data/results/retrieval_results.jsonl`: 질의·방법별 top-10 순위와 hit
- `data/results/retrieval_metrics.*`: Hit/Recall/MRR/nDCG
- `data/results/retrieval_metrics_hard.*`: Hard split(CVE ID 미포함) 방법별
  Hit/Recall/MRR/nDCG, CVE-weighted(400행, 중복 질의 포함)
- `data/results/retrieval_metrics_hard_query_weighted.*`: 위와 동일하지만
  (question_type, question_ko) 기준 dedup한 307개 unique 질의로 집계
- `data/results/retrieval_metrics_hard_by_type.*`,
  `retrieval_metrics_hard_by_type_query_weighted.*`: 템플릿(난이도)별 세부 지표,
  CVE-weighted/query-weighted 각각
- `data/results/generated_answers_qwen.jsonl`: 모델 원문, 복구 JSON, 파싱 상태,
  인용 및 오류. JSON 실패 행도 삭제하지 않는다. 각 행에는 생성 모델명과
  generation 설정 metadata를 함께 기록한다.
- `data/results/stress_generated_answers_qwen.jsonl`: 답변 불가능 조건의 Qwen 출력
- `data/results/evidence_gate_metrics.*`: Evidence Gate 수용률, 오수용률, 선택적 위험 지표
- `data/results/stress_abstention_metrics.*`: 스트레스 유형별 abstention/gating 지표
- `data/results/answer_auto_metrics_qwen.*`: 방법별 자동 신뢰성 지표
- `data/results/human_eval_template_qwen.csv`: 0–2점 수기 평가 양식
- `data/results/human_eval_sample_qwen.csv`: 방법명을 가린 paired human 평가 표본
- `data/results/human_eval_sample_key_qwen.csv`: 표본의 방법 매핑 키(평가자에게 비공개)
- `data/results/table1_*.{csv,md}` ~ `table4_*.{csv,md}`: 논문용 표

전체 모델 출력까지 검토해야 하는 경우에는 GitHub 본문에 직접 커밋하기보다 release
artifact나 별도 압축 파일로 보관하는 편이 낫다.

Human rubric은 다음과 같다. Evidence Faithfulness는 핵심 주장이 모두 근거에
있으면 2, 일부만 분명하면 1, 핵심 주장이 근거와 다르거나 없으면 0이다. Citation
Correctness는 직접 근거 2, 관련은 있으나 약함 1, 잘못되거나 무관함 0이다.
Hallucination은 없음 0, 경미한 비근거 주장 1, 중요한 오류/비근거 주장 2이다.
수기 열을 채운 뒤 `make_tables.py`를 재실행하면 유효한 human 점수만 Table 3에
추가된다.

방법 4개 × 질문 유형 5개 × 10개 QA의 paired 표본 200건은 다음과 같이 만든다.

```bash
python3 -m src.human_eval_sample prepare
```

평가자는 `human_eval_sample_qwen.csv`만 열어 0–2점 열을 채운다. 방법 노출을
막기 위해 `human_eval_sample_key_qwen.csv`는 평가자에게 제공하지 않는다.
평가 완료 후 점수를 전체 template에 병합하고 표를 다시 만든다.

```bash
python3 -m src.human_eval_sample merge
python3 -m src.make_tables
```

ChatGPT 등 별도의 모델이 평가한 결과는 human 점수와 섞지 않는다. 평가 결과를
`data/results/ai_judge_eval_qwen.csv`로 저장한 뒤 다음 명령으로 방법별·질문
유형별 LLM-as-a-Judge 표를 생성한다.

```bash
python3 -m src.evaluate_ai_judge
```

## Evidence-Gated RAG 스트레스 실험

기존 500개 in-domain QA 결과는 변경하지 않고, 다음의 답변 불가능 조건을 균형
표본으로 추가한다.

- 현재 100개 문서 코퍼스에 없는 실제 held-out CVE: 100 QA
- 존재하지 않는 미래 CVE 식별자: 50 QA
- 정답 CVE 문서를 top-k context에서 제거한 evidence ablation: 50 QA

총 200개 스트레스 질의를 4개 검색 전략으로 평가하고 Qwen 답변 800개를 별도
생성한다. 생성은 10건마다 캐시되므로 같은 명령으로 재개할 수 있다.

```bash
CUDA_VISIBLE_DEVICES=1 python3 scripts/run_reliability_stress.py
```

결과는 `evidence_gate_metrics.{csv,md}`,
`stress_abstention_metrics.{csv,md}` 및 `evidence_gate_decisions.csv`에 저장된다.
Evidence Gate는 질문의 CVE ID가 top-5 근거에 존재하는지, JSON 파싱에
성공했는지, 생성 답변이 동일 CVE 문서를 실제로 인용했는지를 순차적으로
검증한다.

## 논문 Method 섹션용 문장

본 연구는 NVD CVE와 CISA Known Exploited Vulnerabilities 자료를 결합한 뒤,
CVSS v3.x 필드 완전성, 심각도 분포와 CWE 다양성을 고려하여 100개 CVE를 선정하였다. 각 CVE에
대해 다섯 유형의 한국어 질문을 결정적 템플릿으로 생성하여 총 500개 질의를
구성하였다. 동일 코퍼스에서 BM25, BGE-M3 dense retrieval, min-max 정규화
가중합 hybrid retrieval, BGE reranker를 적용한 hybrid retrieval을 비교했으며,
검색 성능은 Hit@1, Recall@5/10, MRR@10, nDCG@10으로 평가하였다. 생성 단계에는
각 전략의 상위 5개 문서를 동일한 Qwen/Qwen2.5-7B-Instruct 프롬프트에 제공하고,
구조화 필드 일치도, CWE F1, 인용 정확도 및 수기 충실도를 측정하였다. 질문에
포함된 CVE ID 자체가 검색에 미치는 lexical shortcut 효과를 분리하기 위해,
동일한 100개 CVE에 대해 CVE ID를 포함하지 않고 제품·버전·CVSS 공격 조건만으로
구성된 400개 질의(hard split)를 추가로 평가하였다.

## 실험 한계

- 고정 템플릿 질의는 실제 사용자 표현의 다양성을 완전히 반영하지 않는다. Hard
  split은 CVE ID라는 lexical shortcut을 제거했을 뿐, 여전히 결정적 템플릿이며
  실제 사용자의 자유 형식 질의를 대체하지 않는다.
- NVD 설명과 CPE는 불완전할 수 있고 reference URL 자체는 페이지 본문 근거가 아니다.
- 한 CVE당 하나의 통합 chunk를 사용하는 설계는 긴 외부 advisory의 세부 내용을
  평가하지 않는다.
- Dense/reranker 모델의 다국어 표현 편향과 NVD 수집 시점에 따라 결과가 달라질 수 있다.
- 자동 필드 일치와 인용-CVE 일치는 서술형 답변의 전체 사실성을 대신하지 않으므로
  human evaluation을 함께 보고해야 한다.
