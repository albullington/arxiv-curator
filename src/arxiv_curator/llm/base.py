from typing import Protocol

from arxiv_curator.models import Paper


class Summarizer(Protocol):
    def summarize(self, paper: Paper) -> str: ...


class Explainer(Protocol):
    def explain(self, paper: Paper, interest_profile_text: str, signals: dict) -> str: ...
