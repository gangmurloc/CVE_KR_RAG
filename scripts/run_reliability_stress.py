from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_stage(module: str, config: str | None, force: bool = False) -> None:
    command = [sys.executable, "-m", module]
    if config:
        command.extend(["--config", str(Path(config).resolve())])
    if force and module not in {"src.evaluate_evidence_gate"}:
        command.append("--force")
    print(f"\n[stress stage] {module}", flush=True)
    try:
        subprocess.run(command, cwd=ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"[stress failed] {module} (exit code {exc.returncode})"
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evidence-Gated RAG reliability stress experiment"
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for module in (
        "src.build_stress_dataset",
        "src.retrieval_stress",
        "src.generate_stress_answers_qwen",
        "src.evaluate_evidence_gate",
    ):
        run_stage(module, args.config, args.force)
    print(
        f"\n[stress done] results={ROOT / 'data' / 'results' / 'evidence_gate_metrics.csv'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
