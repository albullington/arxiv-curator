import numpy as np
import pytest

from arxiv_curator.models import Feedback
from arxiv_curator.rank import (
    feedback_weight, compute_centroids, score_paper,
    overlapping_keywords, most_similar_liked,
)


def test_feedback_weight_full_read_is_one():
    fb = Feedback(arxiv_id="1", created_at="t", rating="up", pages_read=12, total_pages=12)
    assert feedback_weight(fb) == pytest.approx(1.0)


def test_feedback_weight_half_read_is_point_seven_five():
    fb = Feedback(arxiv_id="1", created_at="t", rating="up", pages_read=6, total_pages=12)
    assert feedback_weight(fb) == pytest.approx(0.75)


def test_feedback_weight_no_read_depth_is_one():
    fb = Feedback(arxiv_id="1", created_at="t", rating="up")
    assert feedback_weight(fb) == pytest.approx(1.0)


def test_compute_centroids_separates_liked_and_disliked():
    vectors = {
        "liked1": np.array([1.0, 0.0]),
        "disliked1": np.array([0.0, 1.0]),
    }
    feedback_items = [
        Feedback(arxiv_id="liked1", created_at="t", rating="up"),
        Feedback(arxiv_id="disliked1", created_at="t", rating="down"),
    ]
    mean_liked, mean_disliked = compute_centroids(feedback_items, vectors)
    assert list(mean_liked) == [1.0, 0.0]
    assert list(mean_disliked) == [0.0, 1.0]


def test_compute_centroids_with_no_feedback_returns_none():
    mean_liked, mean_disliked = compute_centroids([], {})
    assert mean_liked is None
    assert mean_disliked is None


def test_score_paper_adds_liked_centroid_and_subtracts_disliked():
    paper_vector = np.array([1.0, 0.0])
    interest_vector = np.array([1.0, 0.0])
    mean_liked = np.array([1.0, 0.0])
    mean_disliked = np.array([0.0, 1.0])
    similarity, adjustment, final = score_paper(
        paper_vector, interest_vector, mean_liked, mean_disliked, alpha=0.3, beta=0.3,
    )
    assert similarity == pytest.approx(1.0)
    assert adjustment == pytest.approx(0.3)
    assert final == pytest.approx(1.3)


def test_score_paper_with_no_centroids_is_similarity_only():
    paper_vector = np.array([1.0, 0.0])
    interest_vector = np.array([1.0, 0.0])
    similarity, adjustment, final = score_paper(paper_vector, interest_vector, None, None)
    assert adjustment == 0.0
    assert final == pytest.approx(similarity)


def test_overlapping_keywords_is_case_insensitive():
    abstract = "This paper studies Retrieval-Augmented Generation for agents."
    keywords = ["retrieval", "gardening"]
    assert overlapping_keywords(abstract, keywords) == ["retrieval"]


def test_most_similar_liked_picks_closest_vector():
    paper_vector = np.array([1.0, 0.0])
    liked_vectors = {
        "close": np.array([1.0, 0.1]),
        "far": np.array([0.0, 1.0]),
    }
    assert most_similar_liked(paper_vector, liked_vectors) == "close"


def test_most_similar_liked_with_no_liked_papers_returns_none():
    assert most_similar_liked(np.array([1.0, 0.0]), {}) is None


from arxiv_curator import db, rank
from arxiv_curator.models import Paper


class StubExplainer:
    def explain(self, paper, interest_profile_text, signals):
        return f"stub explanation for {paper.arxiv_id}"


def make_paper(arxiv_id, abstract):
    return Paper(
        arxiv_id=arxiv_id, title=f"Title {arxiv_id}", authors="Author",
        abstract=abstract, categories="cs.AI",
        published="2026-01-01T00:00:00Z", url=f"https://arxiv.org/abs/{arxiv_id}",
    )


def test_rank_papers_scores_relevant_paper_higher(tmp_path, monkeypatch):
    conn = db.get_connection(":memory:")
    db.init_db(conn)
    db.insert_paper(conn, make_paper("relevant1", "This paper is about transformers and deep learning."))
    db.insert_paper(conn, make_paper("irrelevant1", "This paper is about gardening tips."))

    interests_path = tmp_path / "interests.yaml"
    interests_path.write_text("summary: I like transformers and deep learning.\nkeywords:\n  - transformers\n")

    def fake_embed_texts(texts, client):
        import numpy as np
        return np.array([[1.0, 0.0] if "transformer" in t.lower() else [0.0, 1.0] for t in texts])

    monkeypatch.setattr(rank, "embed_texts", fake_embed_texts)

    scores = rank.rank_papers(conn, interests_path, StubExplainer(), client=None)
    scores_by_id = {s.arxiv_id: s for s in scores}
    assert scores_by_id["relevant1"].final_score > scores_by_id["irrelevant1"].final_score
    assert "relevant1" in scores_by_id["relevant1"].explanation
    assert db.get_score(conn, "relevant1") is not None


def test_get_paper_vectors_only_embeds_missing_papers(monkeypatch):
    conn = db.get_connection(":memory:")
    db.init_db(conn)
    cached_paper = make_paper("cached1", "already embedded")
    fresh_paper = make_paper("fresh1", "needs embedding")
    db.insert_paper(conn, cached_paper)
    db.insert_paper(conn, fresh_paper)
    db.upsert_embedding(conn, "cached1", [1.0, 1.0])

    embed_calls = []

    def fake_embed_texts(texts, client):
        embed_calls.append(texts)
        return np.array([[0.0, 1.0] for _ in texts])

    monkeypatch.setattr(rank, "embed_texts", fake_embed_texts)

    vectors_by_id = rank.get_paper_vectors(conn, [cached_paper, fresh_paper], client=None)

    assert list(vectors_by_id["cached1"]) == [1.0, 1.0]
    assert list(vectors_by_id["fresh1"]) == [0.0, 1.0]
    assert embed_calls == [["needs embedding"]]
    assert list(db.get_embeddings(conn, ["fresh1"])["fresh1"]) == [0.0, 1.0]


def test_rank_papers_reuses_existing_explanation_on_rerank(tmp_path, monkeypatch):
    conn = db.get_connection(":memory:")
    db.init_db(conn)
    db.insert_paper(conn, make_paper("cached1", "This paper is about transformers."))

    interests_path = tmp_path / "interests.yaml"
    interests_path.write_text("summary: I like transformers.\n")

    def fake_embed_texts(texts, client):
        return np.array([[1.0, 0.0] for _ in texts])

    monkeypatch.setattr(rank, "embed_texts", fake_embed_texts)

    class CountingExplainer:
        def __init__(self):
            self.calls = 0

        def explain(self, paper, interest_profile_text, signals):
            self.calls += 1
            return f"explanation #{self.calls} for {paper.arxiv_id}"

    explainer = CountingExplainer()

    rank.rank_papers(conn, interests_path, explainer, client=None)
    assert explainer.calls == 1
    first_explanation = db.get_score(conn, "cached1").explanation

    rank.rank_papers(conn, interests_path, explainer, client=None)
    assert explainer.calls == 1
    assert db.get_score(conn, "cached1").explanation == first_explanation


def test_rank_papers_explains_only_newly_added_papers(tmp_path, monkeypatch):
    conn = db.get_connection(":memory:")
    db.init_db(conn)
    db.insert_paper(conn, make_paper("old1", "This paper is about transformers."))

    interests_path = tmp_path / "interests.yaml"
    interests_path.write_text("summary: I like transformers.\n")

    def fake_embed_texts(texts, client):
        return np.array([[1.0, 0.0] for _ in texts])

    monkeypatch.setattr(rank, "embed_texts", fake_embed_texts)

    class CountingExplainer:
        def __init__(self):
            self.calls = 0

        def explain(self, paper, interest_profile_text, signals):
            self.calls += 1
            return f"explanation #{self.calls}"

    explainer = CountingExplainer()
    rank.rank_papers(conn, interests_path, explainer, client=None)
    assert explainer.calls == 1

    db.insert_paper(conn, make_paper("new1", "Another transformers paper."))
    rank.rank_papers(conn, interests_path, explainer, client=None)
    assert explainer.calls == 2
