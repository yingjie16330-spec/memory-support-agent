"""Project configuration helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

load_dotenv(PROJECT_ROOT / ".env", override=False)


@dataclass
class Config:
    """Simple configuration object for the demo project."""

    openai_api_key: str = os.getenv("OPENAI_API_KEY", "").strip()
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.berget.ai/v1").strip()
    openai_model: str = os.getenv("OPENAI_MODEL", "gemma-4-31B-it").strip()
    kb_top_k: int = 3
    memory_top_k: int = 3
    kb_score_threshold: float = 1.0


def get_config() -> Config:
    """Return project configuration using environment variables when available."""

    return Config()
