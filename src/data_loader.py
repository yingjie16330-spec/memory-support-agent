"""Utilities for loading the synthetic JSONL datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config import DATA_DIR


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dictionaries."""

    file_path = Path(path)
    records: list[dict[str, Any]] = []

    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    return records


def load_knowledge_base() -> list[dict[str, Any]]:
    return load_jsonl(DATA_DIR / "knowledge_base.jsonl")


def load_user_memory() -> list[dict[str, Any]]:
    return load_jsonl(DATA_DIR / "user_memory.jsonl")


def load_test_set() -> list[dict[str, Any]]:
    return load_jsonl(DATA_DIR / "test_set.jsonl")
