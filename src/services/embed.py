import os
import threading

from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "multi-qa-mpnet-base-dot-v1"

_model: SentenceTransformer | None = None
_lock = threading.Lock()


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                model_name = os.environ.get("YOURMEMORY_EMBED_MODEL", DEFAULT_MODEL)
                _model = SentenceTransformer(model_name)
    return _model


def embed(text: str) -> list[float]:
    return _get_model().encode(text).tolist()
