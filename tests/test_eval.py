import json

import numpy as np
import pytest

from arxiv_curator import db, eval as eval_module, rank
from arxiv_curator.models import Feedback, Paper
from arxiv_curator.eval import precision_at_k, ndcg_at_k, mrr, evaluate, append_history


def test_precision_at_k_counts_hits_in_top_k():
    ranked = ["a", "b", "c", "d"]
    relevant = {"a", "c"}
    assert precision_at_k(ranked, relevant, 2) == pytest.approx(0.5)


def test_precision_at_k_empty_ranked_is_zero():
    assert precision_at_k([], {"a"}, 5) == 0.0


def test_mrr_returns_reciprocal_rank_of_first_hit():
    ranked = ["a", "b", "c"]
    assert mrr(ranked, {"c"}) == pytest.approx(1 / 3)


def test_mrr_with_no_hit_is_zero():
    assert mrr(["a", "b"], {"z"}) == 0.0


def test_ndcg_at_k_perfect_ranking_is_one():
    ranked = ["a", "b", "c"]
    relevant = {"a"}
    assert ndcg_at_k(ranked, relevant, 3) == pytest.approx(1.0)


def test_ndcg_at_k_relevant_item_lower_in_ranking_scores_less_than_one():
    ranked = ["b", "a", "c"]
    relevant = {"a"}
    score = ndcg_at_k(ranked, relevant, 3)
    assert 0 < score < 1.0


def test_evaluate_reports_insufficient_data_below_threshold():
    feedback_items = [Feedback(arxiv_id=str(i), created_at="t", rating="up") for i in range(3)]
    result = evaluate(feedback_items, vectors_by_id={}, interest_vector=np.array([1.0, 0.0]))
    assert result["status"] == "insufficient_data"
    assert result["rated_count"] == 3


def test_evaluate_returns_metrics_with_enough_feedback():
    vectors_by_id = {
        str(i): (np.array([1.0, 0.0]) if i % 2 == 0 else np.array([0.0, 1.0]))
        for i in range(6)
    }
    feedback_items = [
        Feedback(arxiv_id=str(i), created_at="t", rating="up" if i % 2 == 0 else "down")
        for i in range(6)
    ]
    interest_vector = np.array([1.0, 0.0])
    result = evaluate(feedback_items, vectors_by_id, interest_vector)
    assert result["status"] == "ok"
    assert "feedback_adjusted" in result
    assert "similarity_only_baseline" in result
    assert "random_baseline" in result
    assert 0.0 <= result["feedback_adjusted"]["precision_at_5"] <= 1.0
    assert "precision_at_10" in result["similarity_only_baseline"]
    assert "precision_at_10" in result["random_baseline"]


def test_evaluate_ok_result_includes_rated_count_and_n_papers():
    vectors_by_id = {
        str(i): (np.array([1.0, 0.0]) if i % 2 == 0 else np.array([0.0, 1.0]))
        for i in range(6)
    }
    feedback_items = [
        Feedback(arxiv_id=str(i), created_at="t", rating="up" if i % 2 == 0 else "down")
        for i in range(6)
    ]
    result = evaluate(feedback_items, vectors_by_id, interest_vector=np.array([1.0, 0.0]))
    assert result["status"] == "ok"
    assert result["rated_count"] == 6
    assert result["n_papers"] == 6


def test_run_eval_wires_db_and_embeddings(tmp_path, monkeypatch):
    conn = db.get_connection(":memory:")
    db.init_db(conn)
    for i in range(6):
        db.insert_paper(conn, Paper(
            arxiv_id=str(i), title=f"T{i}", authors="A", abstract="transformers" if i % 2 == 0 else "gardening",
            categories="cs.AI", published="2026-01-01T00:00:00Z", url=f"https://arxiv.org/abs/{i}",
        ))
        from arxiv_curator import feedback as feedback_module
        feedback_module.record_feedback(conn, str(i), rating="up" if i % 2 == 0 else "down")

    interests_path = tmp_path / "interests.yaml"
    interests_path.write_text("summary: I like transformers.\n")

    def fake_embed_texts(texts, client):
        return np.array([[1.0, 0.0] if "transformer" in t.lower() else [0.0, 1.0] for t in texts])

    monkeypatch.setattr(eval_module, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(rank, "embed_texts", fake_embed_texts)
    result = eval_module.run_eval(conn, interests_path, client=None)
    assert result["status"] == "ok"


def _ok_result():
    metric_block = {"precision_at_5": 0.08, "precision_at_10": 0.06, "ndcg_at_10": 0.249, "mrr": 0.146}
    return {
        "status": "ok",
        "n_evaluated": 5,
        "rated_count": 5,
        "n_papers": 312,
        "feedback_adjusted": dict(metric_block),
        "similarity_only_baseline": dict(metric_block),
        "random_baseline": dict(metric_block),
    }


def test_append_history_writes_parseable_line_with_timestamp(tmp_path):
    path = append_history(_ok_result(), tmp_path)
    assert path == tmp_path / "eval_history.jsonl"
    lines = path.read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["rated_count"] == 5
    assert record["n_papers"] == 312
    assert record["feedback_adjusted"]["ndcg_at_10"] == 0.249
    assert "status" not in record
    assert record["timestamp"].endswith("+00:00")


def test_append_history_appends_rather_than_overwrites(tmp_path):
    append_history(_ok_result(), tmp_path)
    append_history(_ok_result(), tmp_path)
    lines = (tmp_path / "eval_history.jsonl").read_text().splitlines()
    assert len(lines) == 2
