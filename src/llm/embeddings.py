import numpy as np

from arxiv_curator.llm.retry import with_retries

EMBEDDING_MODEL = "gemini-embedding-001"


def embed_texts(texts: list[str], client) -> np.ndarray:
    if not texts:
        return np.empty((0, 0))
    result = with_retries(client.models.embed_content, model=EMBEDDING_MODEL, contents=texts)
    return np.array([embedding.values for embedding in result.embeddings])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
