"""Local embedding layer.

Wraps a sentence-transformers model so the rest of the pipeline can turn
submission text into unit-length vectors without touching the model directly.
Embeddings are cached by text so repeated scoring stays fast.
"""

from __future__ import annotations

import threading
from functools import lru_cache

import numpy as np

_MODEL_NAME = "all-MiniLM-L6-v2"
_model = None
_model_lock = threading.Lock()


def _get_model():
    """Load the model once, lazily (import is heavy)."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer

                _model = SentenceTransformer(_MODEL_NAME)
    return _model


@lru_cache(maxsize=4096)
def _embed_one(text: str) -> tuple[float, ...]:
    vec = _get_model().encode(text, normalize_embeddings=True)
    return tuple(float(x) for x in vec)


def embed(texts: list[str]) -> np.ndarray:
    """Return an (n, d) array of unit-length embeddings for the given texts."""
    return np.array([_embed_one(t) for t in texts], dtype=np.float64)


def embed_submission(headline: str, body: str) -> np.ndarray:
    """Embed a single submission's headline + body as one vector."""
    return embed([f"{headline}. {body}"])[0]
