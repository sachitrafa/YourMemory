import os
import threading

# Set YOURMEMORY_EMBED_BACKEND=spacy to use spaCy en_core_web_md vectors (300-dim)
# instead of the default sentence-transformers model (768-dim).
# Use this on networks where HuggingFace is blocked.
# NOTE: switching backends requires a fresh database — dimensions are incompatible.
BACKEND = os.environ.get("YOURMEMORY_EMBED_BACKEND", "sentence_transformers").lower()

DEFAULT_MODEL = "multi-qa-mpnet-base-dot-v1"

_model = None
_lock = threading.Lock()


def _get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                if BACKEND == "spacy":
                    try:
                        import spacy
                        _model = spacy.load("en_core_web_md")
                    except OSError:
                        raise RuntimeError(
                            "spaCy model 'en_core_web_md' not found. "
                            "Run: python -m spacy download en_core_web_md"
                        ) from None
                else:
                    from sentence_transformers import SentenceTransformer
                    model_name = os.environ.get("YOURMEMORY_EMBED_MODEL", DEFAULT_MODEL)
                    _model = SentenceTransformer(model_name)
    return _model


def embed(text: str) -> list[float]:
    model = _get_model()
    if BACKEND == "spacy":
        return model(text).vector.tolist()
    return model.encode(text).tolist()
