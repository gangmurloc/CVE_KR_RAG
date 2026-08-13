# 한국어 CVE RAG 검색 전략별 신뢰성 평가

## Abstract (English)

A reproducible evaluation pipeline for Korean-language CVE question
answering under four retrieval strategies — BM25, dense retrieval
(BAAI/bge-m3), hybrid (BM25+dense), and hybrid with a cross-encoder
reranker (BAAI/bge-reranker-v2-m3) — measuring both retrieval quality and
downstream answer reliability with a fixed generation model
(Qwen/Qwen2.5-7B-Instruct). Data is collected live from the NVD CVE API and
the CISA Known Exploited Vulnerabilities catalog; the pipeline does not
fabricate or hand-tune any reported number.

**Main finding.** On the original 500-question benchmark, BM25 reaches
Hit@1 = 1.000, which looks like an outright win — but every question embeds
the gold CVE ID verbatim, and the BM25 tokenizer explicitly up-weights CVE
ID tokens, so this measures lexical exact-match on a unique identifier, not
retrieval skill. To isolate the real effect, this repo adds a second,
400-question **hard split** that never mentions the CVE ID and instead
describes each vulnerability only by product/vendor/version, CVSS attack
conditions (attack vector, privileges required, user interaction), and
severity/impact. Once the identifier is removed, BM25's Hit@1 drops to
0.6700 and **Hybrid becomes the best method on Hit@1 (0.7050)**, with
Hybrid+Reranker best on Recall@5/10 — a different ranking from the easy
split, where BM25 dominates on every metric.

Some hard-split templates strip enough information that more than one CVE
in the 100-document corpus can produce byte-identical question text (most
acutely for `attribute_only_hard`, which drops the product name entirely —
34 of 100 CVEs share one attack-vector/privileges/user-interaction/impact
profile). Scoring such a question against only the one CVE that happened to
generate it would count a semantically correct retrieval as a miss whenever
an indistinguishable sibling exists, so retrieval is instead scored against
the **set** of all CVEs whose deterministic template output matches the
question verbatim (`gold_cve_id_set` in the data; 115 of 400 hard questions
have more than one gold CVE). See "Hard split" below for the per-template
breakdown and why this matters most for `attribute_only_hard`.

| Split | Method | Hit@1 | MRR@10 |
| --- | --- | --- | --- |
| Easy (500, CVE ID in question) | BM25 | 1.0000 | 1.0000 |
| Easy (500, CVE ID in question) | Hybrid + Reranker | 0.9940 | 0.9967 |
| Hard (400, no CVE ID, gold-set scored) | BM25 | 0.6700 | 0.7019 |
| Hard (400, no CVE ID, gold-set scored) | Hybrid | 0.7050 | 0.7447 |

Full tables (retrieval, answer accuracy, evidence-gated abstention,
per-template hard-split breakdown) are under `data/results/`. Setup
instructions are in `REPRODUCIBILITY.md`. The rest of this README, including
the full Method/Limitations sections, is in Korean.

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
순위를 `rank_of_gold_cve`로 사용한다 — 단일 CVE가 아니라 집합 전체를 기준으로
Hit@1/Recall/MRR을 계산한다. 400개 hard 질문 중 115개(`gold_set_size > 1`)가
실제로 둘 이상의 정답을 가지며, 그중 34개는 `attribute_only_hard`에서 하나의
공격 조건·영향도 조합을 공유하는 최대 그룹이다.

**Hard split 결과 (400 질의, CVE ID 미포함, gold set 기준 채점):**

| Method | Hit@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
| --- | --- | --- | --- | --- | --- |
| BM25 | 0.6700 | 0.7475 | 0.7650 | 0.7019 | 0.7172 |
| Dense (BGE-M3) | 0.6975 | 0.7775 | 0.7925 | 0.7305 | 0.7456 |
| Hybrid | 0.7050 | 0.8100 | 0.8325 | 0.7447 | 0.7659 |
| Hybrid + Reranker | 0.6500 | 0.8225 | 0.8600 | 0.7332 | 0.7644 |

CVE ID를 제거하자 BM25의 Hit@1은 1.000 → 0.6700으로 떨어지고, Hit@1 기준으로는
Hybrid(0.7050)가, Recall@5/10 기준으로는 Hybrid+Reranker가 가장 높다. 즉 500개
easy 질문에서 관찰된 "BM25가 가장 강하다"는 결과는 검색 전략의 우수성이 아니라
질문에 남아있는 고유 식별자 때문이었다는 진단이 hard split 결과로 다시 확인된다.
다만 easy split과 달리 hard split에서는 한 가지 방법이 모든 지표를 지배하지
않는다 — Hybrid+Reranker는 상위 5/10위 안에는 더 잘 넣지만(Recall), 1위
정확도(Hit@1)는 오히려 Hybrid보다 낮다.

템플릿(난이도)별로 보면 그 원인이 더 분명해진다.

| Method | product_version | severity_scenario | attack_condition | attribute_only |
| --- | --- | --- | --- | --- |
| BM25 | 0.7700 | 0.8000 | 0.7700 | 0.3400 |
| Dense | 0.8600 | 0.7300 | 0.7900 | 0.4100 |
| Hybrid | 0.8000 | 0.8200 | 0.7900 | 0.4100 |
| Hybrid + Reranker | 0.9000 | 0.7500 | 0.8800 | 0.0700 |

(값은 Hit@1) 벤더/제품명이 질문에 남아 있는 세 템플릿에서는 모든 방법이
0.73~0.90 수준을 유지하지만, 벤더/제품명을 완전히 제거하고 공격 조건(AV/PR/UI)과
영향도만 남긴 `attribute_only_hard`에서는 BM25/Dense/Hybrid가 0.34~0.41로 떨어진다
(gold-set 채점 이전에는 0.01~0.02로 계산되었으나, 이는 앞서 설명한 단일-CVE
채점의 함정 때문이었다). 여전히 0.4 안팎이라는 것은 방법론의 결함이 아니라
"제품명 없이는 검색 전략을 아무리 바꿔도 정답 집합 중 하나를 1위로 정확히
찍기가 원래도 어렵다"는, retrieval 자체의 한계를 보여준다.

`attribute_only_hard`에서 Hybrid+Reranker만 Hit@1이 0.0700으로 다른 세 방법
(0.34~0.41)보다 뚜렷하게 낮으면서 Recall@5(0.47)·Recall@10(0.58)는 오히려 가장
높다는 점은 눈에 띄는 이상 패턴이다. 즉 reranker가 정답 집합에 속한 문서를
상위 10위 안에는 잘 포함시키지만, 1위로 정확히 올리는 데는 다른 방법보다 약하다.
정답이 여러 개인 상황에서 cross-encoder가 정답 집합 내부의 특정 CVE 문서
하나에 과도하게 높은 점수를 주기보다 집합 전체에 점수를 고르게 분산시키기
때문일 수 있다는 가설은 있으나, 이 저장소의 실험만으로 원인을 확정할 수는
없다 — 후속 분석이 필요한 관찰로 남겨둔다.

재현:

```bash
python scripts/run_all.py --mode hard_retrieval_only
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
- `data/results/retrieval_metrics_hard.*`: Hard split(CVE ID 미포함) 방법별 Hit/Recall/MRR/nDCG
- `data/results/retrieval_metrics_hard_by_type.*`: Hard split 템플릿(난이도)별 세부 지표
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
