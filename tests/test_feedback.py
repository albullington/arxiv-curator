import pytest
from arxiv_curator import db, feedback
from arxiv_curator.models import Paper


def make_conn_with_paper():
    conn = db.get_connection(":memory:")
    db.init_db(conn)
    db.insert_paper(conn, Paper(
        arxiv_id="2601.00001", title="T", authors="A", abstract="B",
        categories="cs.AI", published="2026-01-01T00:00:00Z",
        url="https://arxiv.org/abs/2601.00001",
    ))
    return conn


def test_record_feedback_with_rating():
    conn = make_conn_with_paper()
    feedback.record_feedback(conn, "2601.00001", rating="up")
    items = db.list_feedback(conn)
    assert items[0].rating == "up"


def test_record_feedback_with_read_depth_only():
    conn = make_conn_with_paper()
    feedback.record_feedback(conn, "2601.00001", pages_read=5, total_pages=10)
    items = db.list_feedback(conn)
    assert items[0].pages_read == 5
    assert items[0].rating is None


def test_record_feedback_rejects_invalid_rating():
    conn = make_conn_with_paper()
    with pytest.raises(ValueError, match="rating"):
        feedback.record_feedback(conn, "2601.00001", rating="sideways")


def test_record_feedback_rejects_unknown_paper():
    conn = db.get_connection(":memory:")
    db.init_db(conn)
    with pytest.raises(ValueError, match="No such paper"):
        feedback.record_feedback(conn, "9999.00000", rating="up")


def test_record_feedback_rejects_pages_read_exceeding_total():
    conn = make_conn_with_paper()
    with pytest.raises(ValueError, match="pages_read"):
        feedback.record_feedback(conn, "2601.00001", pages_read=20, total_pages=10)


def test_record_feedback_rejects_empty_feedback():
    conn = make_conn_with_paper()
    with pytest.raises(ValueError, match="rating, note, or pages_read"):
        feedback.record_feedback(conn, "2601.00001")
