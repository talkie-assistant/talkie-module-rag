"""
Ollama embedding client for RAG: POST /api/embed, optional model check via /api/tags.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

# Fallback embedding models to try if configured model is not installed
EMBEDDING_MODEL_FALLBACKS = [
    "nomic-embed-text",
    "mxbai-embed-large",
    "all-minilm",
]

PULL_MESSAGE = "Pull an embedding model: ollama pull nomic-embed-text"


def _get_available_models(base_url: str, timeout_sec: float = 10.0) -> set[str]:
    """Return set of model names available on Ollama (from /api/tags)."""
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=timeout_sec)
        if r.status_code != 200:
            return set()
        data = r.json()
        models = data.get("models") or []
        return {m.get("name", "").split(":")[0] for m in models if m.get("name")}
    except requests.RequestException:
        return set()


def resolve_embedding_model(
    base_url: str, configured: str, timeout_sec: float = 10.0
) -> str:
    """
    Return an embedding model name that is available. Tries configured first, then fallbacks.
    Raises ValueError with a copy-pasteable message if none available.
    """
    available = _get_available_models(base_url, timeout_sec)
    if not available:
        raise ValueError(
            "Ollama not reachable or returned no models. Ensure Ollama is running. "
            + PULL_MESSAGE
        )
    configured_base = (configured or "").strip().split(":")[0]
    if configured_base and configured_base in available:
        return configured_base
    for candidate in EMBEDDING_MODEL_FALLBACKS:
        if candidate in available:
            logger.info(
                "Using embedding model %s (configured %s not found)",
                candidate,
                configured,
            )
            return candidate
    raise ValueError(PULL_MESSAGE)


class OllamaEmbedClient:
    """Call Ollama /api/embed; supports single string or list of strings."""

    def __init__(
        self,
        base_url: str,
        model_name: str,
        timeout_sec: float = 120.0,
        max_retries: int = 2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self._model_resolved: str | None = None

    def ensure_model(self) -> str:
        """Resolve and cache embedding model; raise ValueError if none available."""
        if self._model_resolved is not None:
            return self._model_resolved
        self._model_resolved = resolve_embedding_model(self.base_url, self.model_name)
        return self._model_resolved

    def embed(self, inputs: str | list[str]) -> list[list[float]]:
        """
        Get embeddings for one or more texts. Returns list of float vectors.
        On HTTP error (e.g. 404 model not found) raises or returns empty and logs.
        """
        model = self.ensure_model()
        if isinstance(inputs, str):
            inputs = [inputs]
        inputs = [s for s in inputs if s is not None and isinstance(s, str)]
        if not inputs:
            return []

        url = f"{self.base_url}/api/embed"
        payload: dict[str, Any] = {
            "model": model,
            "input": inputs if len(inputs) > 1 else inputs[0],
        }

        for attempt in range(self.max_retries + 1):
            try:
                r = requests.post(url, json=payload, timeout=self.timeout_sec)
                if r.status_code == 404:
                    self._model_resolved = None
                    raise ValueError(PULL_MESSAGE)
                r.raise_for_status()
                data = r.json()
                embs = data.get("embeddings")
                if isinstance(embs, list) and len(embs) == len(inputs):
                    return embs
                logger.warning("Ollama embed returned unexpected shape")
                return []
            except requests.RequestException as e:
                logger.warning("Ollama embed attempt %d failed: %s", attempt + 1, e)
                if attempt < self.max_retries:
                    time.sleep(1.0)
                else:
                    raise
        return []
