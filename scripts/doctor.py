from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "README.md",
    "LICENSE",
    "config.yaml",
    "requirements.txt",
    ".env.example",
    "scripts/run_all.py",
    "scripts/run_reliability_stress.py",
    "src/fetch_nvd.py",
    "src/fetch_cisa_kev.py",
    "src/select_cves.py",
    "src/build_qa_dataset.py",
    "src/build_corpus.py",
    "src/retrieval.py",
    "src/evaluate_retrieval.py",
    "src/generate_answers_qwen.py",
    "src/evaluate_answers.py",
    "src/make_tables.py",
]

REQUIRED_MODULES = [
    "numpy",
    "pandas",
    "requests",
    "yaml",
    "sklearn",
    "rank_bm25",
    "sentence_transformers",
    "torch",
    "transformers",
    "dotenv",
]


def ok(message: str) -> None:
    print(f"[OK] {message}")


def warn(message: str) -> None:
    print(f"[WARN] {message}")


def fail(message: str) -> None:
    print(f"[FAIL] {message}")


def check_files() -> list[str]:
    errors: list[str] = []
    for relative in REQUIRED_FILES:
        path = ROOT / relative
        if path.exists():
            ok(f"found {relative}")
        else:
            errors.append(f"missing {relative}")
            fail(f"missing {relative}")
    return errors


def check_config() -> list[str]:
    errors: list[str] = []
    config_path = ROOT / "config.yaml"
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f"config.yaml cannot be parsed: {exc}")
        return [str(exc)]

    for key in ["experiment", "retrieval", "generation", "fetch", "paths"]:
        if key not in config:
            errors.append(f"config.yaml missing top-level key: {key}")
            fail(f"config.yaml missing {key}")
    if not errors:
        ok("config.yaml parsed")
    return errors


def check_modules() -> list[str]:
    missing: list[str] = []
    for module in REQUIRED_MODULES:
        if importlib.util.find_spec(module) is None:
            missing.append(module)
        else:
            ok(f"python module available: {module}")
    if missing:
        warn("missing optional/runtime modules: " + ", ".join(missing))
    return []


def check_jsonl_samples() -> list[str]:
    errors: list[str] = []
    for relative in [
        "data/processed/selected_cves_100.jsonl",
        "data/processed/qa_ko_500.jsonl",
        "data/corpus/cve_corpus.jsonl",
    ]:
        path = ROOT / relative
        if not path.exists():
            warn(f"sample data not present: {relative}")
            continue
        try:
            first = next(line for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
            json.loads(first)
            ok(f"valid JSONL sample: {relative}")
        except Exception as exc:
            errors.append(f"{relative}: {exc}")
            fail(f"invalid JSONL sample: {relative}: {exc}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Check repository readiness.")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as failures.")
    args = parser.parse_args()

    errors: list[str] = []
    errors.extend(check_files())
    errors.extend(check_config())
    check_modules()
    errors.extend(check_jsonl_samples())

    env_path = ROOT / ".env"
    if env_path.exists():
        warn(".env exists locally and is ignored; do not commit secrets")
    else:
        ok(".env is absent")

    if errors:
        print("\nRepository check failed.")
        for error in errors:
            print(f"- {error}")
        sys.exit(1)

    print("\nRepository check passed.")


if __name__ == "__main__":
    main()
