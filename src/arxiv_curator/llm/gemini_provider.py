from arxiv_curator.models import Paper

DEFAULT_MODEL = "gemini-2.5-flash"


def build_summarize_prompt(paper: Paper) -> str:
    return (
        "Summarize this arXiv abstract in 2-3 plain-English sentences for a "
        "researcher deciding whether to read the full paper.\n\n"
        f"Title: {paper.title}\n\nAbstract: {paper.abstract}"
    )


def build_explain_prompt(paper: Paper, interest_profile_text: str, signals: dict) -> str:
    return (
        "Given this reader's interest profile and the following grounded signals, "
        "write one or two sentences explaining why this paper matches their interests. "
        "Only use the signals given -- don't invent connections that aren't supported by them.\n\n"
        f"Interest profile:\n{interest_profile_text}\n\n"
        f"Paper title: {paper.title}\n"
        f"Overlapping keywords: {signals.get('overlapping_keywords')}\n"
        f"Most similar previously liked paper: {signals.get('most_similar_liked')}\n"
    )


class GeminiProvider:
    def __init__(self, client, model: str = DEFAULT_MODEL):
        self._client = client
        self._model = model

    def summarize(self, paper: Paper) -> str:
        prompt = build_summarize_prompt(paper)
        response = self._client.models.generate_content(model=self._model, contents=prompt)
        return response.text.strip()

    def explain(self, paper: Paper, interest_profile_text: str, signals: dict) -> str:
        prompt = build_explain_prompt(paper, interest_profile_text, signals)
        response = self._client.models.generate_content(model=self._model, contents=prompt)
        return response.text.strip()
