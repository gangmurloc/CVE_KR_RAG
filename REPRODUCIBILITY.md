# Reproducibility Notes

This repository evaluates Korean CVE question answering under four retrieval
settings: BM25, dense retrieval, hybrid retrieval, and hybrid retrieval with a
reranker. The generation model is fixed to `Qwen/Qwen2.5-7B-Instruct` to keep
the comparison focused on retrieval strategy rather than model choice.

## Environment

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/doctor.py
```

For 4-bit Qwen generation, install `bitsandbytes` separately when the local GPU
environment supports it.

## Data Sources

- NVD CVE API 2.0
- CISA Known Exploited Vulnerabilities catalog

`NVD_API_KEY` is optional. Without it, the fetcher uses a slower public-rate
request interval. The project does not require committing API keys or Hugging
Face tokens.

## Standard Runs

Retrieval-only evaluation:

```bash
python scripts/run_all.py --mode retrieval_only
```

Full retrieval, Qwen generation, answer evaluation, and table generation:

```bash
python scripts/run_all.py --mode full_qwen
```

Generation/evaluation from existing retrieval outputs:

```bash
python scripts/run_all.py --mode generation_only
```

Evidence-gated reliability stress evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_reliability_stress.py
```

## Repository Data Policy

Small deterministic datasets and aggregate result tables may be committed for
review and paper reproducibility. Regenerable raw JSONL dumps, dense embedding
caches, full model outputs, human-evaluation working files, and logs are ignored
by default.

If exact generated-answer auditing is required, archive those files separately
or attach them to a release artifact instead of committing them to the main
source tree.
