from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_stage(script: str, config: str | None, force: bool = False) -> None:
    module = f"src.{Path(script).stem}"
    command = [sys.executable, "-m", module]
    if config:
        command.extend(["--config", str(Path(config).resolve())])
    if force and script not in {"evaluate_retrieval.py", "evaluate_answers.py", "make_tables.py"}:
        command.append("--force")
    print(f"\n[stage] {script}", flush=True)
    try:
        subprocess.run(command, cwd=ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"[failed] {script} (exit code {exc.returncode})") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="CVE Korean RAG evaluation pipeline")
    parser.add_argument(
        "--mode", choices=["retrieval_only", "full_qwen", "generation_only"],
        default="retrieval_only",
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    preparation = [
        "fetch_nvd.py", "fetch_cisa_kev.py", "select_cves.py",
        "build_qa_dataset.py", "build_corpus.py", "retrieval.py",
        "evaluate_retrieval.py",
    ]
    if args.mode in {"retrieval_only", "full_qwen"}:
        for script in preparation:
            run_stage(script, args.config, args.force)
    if args.mode in {"full_qwen", "generation_only"}:
        run_stage("generate_answers_qwen.py", args.config, args.force)
        run_stage("evaluate_answers.py", args.config)
    run_stage("make_tables.py", args.config)
    print(f"\n[done] mode={args.mode}; results={ROOT / 'data' / 'results'}", flush=True)


if __name__ == "__main__":
    main()
