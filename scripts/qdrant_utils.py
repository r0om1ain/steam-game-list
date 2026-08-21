from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient

ROOT = Path(__file__).resolve().parents[1]


def load_env():
    load_dotenv(ROOT / ".env")


def collection_name() -> str:
    return os.getenv("QDRANT_COLLECTION", "steam_games")


def get_client(timeout: float | None = None) -> QdrantClient:
    load_env()
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    api_key = os.getenv("QDRANT_API_KEY") or None
    if timeout is None:
        timeout = float(os.getenv("QDRANT_TIMEOUT", "60"))

    if api_key is None and host not in ("localhost", "127.0.0.1"):
        print("warning: pas de QDRANT_API_KEY alors que Qdrant n'est pas en local")

    return QdrantClient(host=host, port=port, api_key=api_key, timeout=timeout)
