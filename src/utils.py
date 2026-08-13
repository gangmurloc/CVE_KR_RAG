from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def setup_logging(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger(name)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else ROOT / "config.yaml"
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config["_root"] = str(config_path.resolve().parent)
    return config


def resolve_path(config: dict[str, Any], key: str) -> Path:
    path = Path(config["paths"][key])
    if not path.is_absolute():
        path = Path(config["_root"]) / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, default=json_default) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value):
        return None
    raise TypeError(f"Cannot serialize {type(value)}")


def as_list(value: Any) -> list[Any]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            return [item.strip() for item in re.split(r"[;,|]", text) if item.strip()]
    return [value]


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def minmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        return scores
    low, high = float(scores.min()), float(scores.max())
    return np.zeros_like(scores) if high == low else (scores - low) / (high - low)


def safe_markdown(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        text = df.to_markdown(index=index)
    except ImportError:
        columns = list(df.columns)
        rows = [["" if pd.isna(v) else str(v) for v in row] for row in df.to_numpy()]
        text = (
            "| " + " | ".join(columns) + " |\n"
            + "| " + " | ".join(["---"] * len(columns)) + " |\n"
            + "\n".join("| " + " | ".join(row) + " |" for row in rows)
        )
    path.write_text(text + "\n", encoding="utf-8")


def extract_json_object(text: str) -> tuple[dict[str, Any] | None, bool]:
    candidates = [text.strip()]
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidates.extend(fenced)
    start = text.find("{")
    if start >= 0:
        depth, in_string, escaped = 0, False, False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : index + 1])
                    break
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed, candidate != text.strip()
        except (json.JSONDecodeError, TypeError):
            continue
    return None, False
