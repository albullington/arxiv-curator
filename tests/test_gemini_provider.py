from types import SimpleNamespace

from arxiv_curator.llm.gemini_provider import (
    build_summarize_prompt, build_explain_prompt, GeminiProvider,
)
from arxiv_curator.models import Paper

PAPER = Paper(
    arxiv_id="2601.00001", title="A Great Paper", authors="Ada Author",
    abstract="This paper studies retrieval-augmented generation.",
    categories="cs.AI", published="2026-01-01T00:00:00Z",
    url="https://arxiv.org/abs/2601.00001",
)


def test_build_summarize_prompt_includes_title_and_abstract():
    prompt = build_summarize_prompt(PAPER)
    assert "A Great Paper" in prompt
    assert "retrieval-augmented generation" in prompt


def test_build_explain_prompt_includes_signals_and_profile():
    signals = {"overlapping_keywords": ["retrieval"], "most_similar_liked": "2500.00002"}
    prompt = build_explain_prompt(PAPER, "I like RAG.", signals)
    assert "I like RAG." in prompt
    assert "retrieval" in prompt
    assert "2500.00002" in prompt


class FakeModels:
    def __init__(self, text):
        self._text = text
        self.last_call = None

    def generate_content(self, model, contents):
        self.last_call = {"model": model, "contents": contents}
        return SimpleNamespace(text=self._text)


class FakeClient:
    def __init__(self, text):
        self.models = FakeModels(text)


def test_provider_summarize_returns_stripped_text():
    client = FakeClient(text="  A short summary.  \n")
    provider = GeminiProvider(client)
    assert provider.summarize(PAPER) == "A short summary."
    assert "A Great Paper" in client.models.last_call["contents"]


def test_provider_explain_returns_stripped_text():
    client = FakeClient(text="  Matches your interests.  ")
    provider = GeminiProvider(client)
    signals = {"overlapping_keywords": [], "most_similar_liked": None}
    assert provider.explain(PAPER, "I like RAG.", signals) == "Matches your interests."
