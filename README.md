# 한국어 CVE RAG 검색 전략별 신뢰성 평가

## 연구 목적

한국어 CVE 보안 질의에 대해 BM25, Dense Retrieval, Hybrid Retrieval,
Hybrid+Reranker가 검색 성능과 최종 답변 신뢰성에 미치는 영향을 동일한
데이터·프롬프트·생성 설정 아래 평가하는 재현 가능한 Python 실험 파이프라인이다.

“본 연구는 생성 모델 성능 비교가 아니라 RAG 검색 전략의 영향을 분석하는 것을 목적으로 하므로, 생성 LLM은 Qwen/Qwen2.5-7B-Instruct로 고정하였다.”

실험은 새 모델을 학습하지 않는다. NVD와 CISA KEV에서 실제 데이터를 수집하며,
코드는 측정되지 않은 수치나 가짜 생성 결과를 만들지 않는다.

## 프로젝트 구조

```text
cve_kr_rag_eval/
├── config.yaml
├── requirements.txt
├── .env.example
├── data/{raw,processed,corpus,results}/
├── src/
│   ├── fetch_nvd.py
│   ├── fetch_cisa_kev.py
│   ├── select_cves.py
│   ├── build_qa_dataset.py
│   ├── build_stress_dataset.py
│   ├── build_corpus.py
│   ├── retrieval.py
│   ├── retrieval_stress.py
│   ├── evaluate_retrieval.py
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
git clone <REPOSITORY_URL>
cd cve_kr_rag_eval
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
- `data/processed/qa_ko_500.*`: CVE당 5개 고정 템플릿 한국어 QA
- `data/corpus/cve_corpus.jsonl`: 검색 문서 및 메타데이터
- `data/results/retrieval_results.jsonl`: 질의·방법별 top-10 순위와 hit
- `data/results/retrieval_metrics.*`: Hit/Recall/MRR/nDCG
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
구조화 필드 일치도, CWE F1, 인용 정확도 및 수기 충실도를 측정하였다.

## 실험 한계

- 고정 템플릿 질의는 실제 사용자 표현의 다양성을 완전히 반영하지 않는다.
- NVD 설명과 CPE는 불완전할 수 있고 reference URL 자체는 페이지 본문 근거가 아니다.
- 한 CVE당 하나의 통합 chunk를 사용하는 설계는 긴 외부 advisory의 세부 내용을
  평가하지 않는다.
- Dense/reranker 모델의 다국어 표현 편향과 NVD 수집 시점에 따라 결과가 달라질 수 있다.
- 자동 필드 일치와 인용-CVE 일치는 서술형 답변의 전체 사실성을 대신하지 않으므로
  human evaluation을 함께 보고해야 한다.
