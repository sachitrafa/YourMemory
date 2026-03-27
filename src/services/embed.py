from sentence_transformers import SentenceTransformer

_model = SentenceTransformer("all-mpnet-base-v2")


def embed(text: str) -> list[float]:
    return _model.encode(text).tolist()
