from types import SimpleNamespace

import numpy as np
import pytest

from arxiv_curator.llm.embeddings import embed_texts, cosine_similarity
from arxiv_curator.llm import factory


class FakeModels:
    def __init__(self, vectors):
        self._vectors = vectors

    def embed_content(self, model, contents):
        return SimpleNamespace(embeddings=[SimpleNamespace(values=v) for v in self._vectors])


class FakeClient:
    def __init__(self, vectors):
        self.models = FakeModels(vectors)


def test_embed_texts_returns_matrix_matching_input_order():
    client = FakeClient(vectors=[[1.0, 0.0], [0.0, 1.0]])
    result = embed_texts(["a", "b"], client)
    assert result.shape == (2, 2)
    assert list(result[0]) == [1.0, 0.0]
    assert list(result[1]) == [0.0, 1.0]


def test_embed_texts_empty_input_returns_empty_array():
    client = FakeClient(vectors=[])
    result = embed_texts([], client)
    assert result.shape[0] == 0


def test_cosine_similarity_identical_vectors_is_one():
    a = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(a, a) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector_is_zero_not_nan():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 0.0])
    assert cosine_similarity(a, b) == 0.0


def test_get_client_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        factory.get_client()


def test_embed_texts_retries_on_transient_error(monkeypatch):
    from google.genai import errors

    monkeypatch.setattr("arxiv_curator.llm.retry.time.sleep", lambda seconds: None)
    calls = {"count": 0}

    class FlakyModels:
        def embed_content(self, model, contents):
            calls["count"] += 1
            if calls["count"] < 2:
                raise errors.APIError(429, {"error": {"message": "rate limited"}})
            return SimpleNamespace(embeddings=[SimpleNamespace(values=[1.0, 0.0]) for _ in contents])

    class FlakyClient:
        def __init__(self):
            self.models = FlakyModels()

    result = embed_texts(["a"], FlakyClient())
    assert list(result[0]) == [1.0, 0.0]
    assert calls["count"] == 2
