from datetime import date

from arxiv_curator import db, digest
from arxiv_curator.models import Paper, Summary, Score


def make_seeded_conn():
    conn = db.get_connection(":memory:")
    db.init_db(conn)
    db.insert_paper(conn, Paper(
        arxiv_id="2601.00001", title="A Great Paper", authors="Ada Author",
        abstract="An abstract.", categories="cs.AI",
        published="2026-01-01T00:00:00Z", url="https://arxiv.org/abs/2601.00001",
    ))
    db.insert_summary(conn, Summary(arxiv_id="2601.00001", text="A short summary.", created_at="t"))
    db.upsert_score(conn, Score(
        arxiv_id="2601.00001", similarity=0.8, feedback_adjustment=0.1, final_score=0.9,
        explanation="Matches your interest in transformers.", created_at="t",
    ))
    return conn


def test_render_digest_shows_pages_when_known():
    conn = make_seeded_conn()
    conn.execute("UPDATE papers SET pages = 27 WHERE arxiv_id = '2601.00001'")
    conn.commit()

    text = digest.render_digest(conn, top_n=20)
    assert "**Length:** 27 pages" in text


def test_render_digest_omits_length_line_when_pages_unknown():
    conn = make_seeded_conn()
    text = digest.render_digest(conn, top_n=20)
    assert "**Length:**" not in text


def test_render_digest_includes_title_summary_score_and_explanation():
    conn = make_seeded_conn()
    text = digest.render_digest(conn, top_n=20)
    assert "A Great Paper" in text
    assert "A short summary." in text
    assert "0.900" in text
    assert "Matches your interest in transformers." in text
    assert "arxiv-curator feedback 2601.00001" in text


def test_render_digest_respects_top_n_and_orders_by_score():
    conn = make_seeded_conn()
    db.insert_paper(conn, Paper(
        arxiv_id="2601.00002", title="Second Paper", authors="A", abstract="B",
        categories="cs.AI", published="2026-01-01T00:00:00Z", url="https://arxiv.org/abs/2601.00002",
    ))
    db.upsert_score(conn, Score(
        arxiv_id="2601.00002", similarity=0.1, feedback_adjustment=0.0, final_score=0.1,
        explanation="Weak match.", created_at="t",
    ))
    text = digest.render_digest(conn, top_n=1)
    assert "A Great Paper" in text
    assert "Second Paper" not in text


def test_write_digest_creates_dated_and_latest_files(tmp_path):
    conn = make_seeded_conn()
    dated_path = digest.write_digest(conn, tmp_path, top_n=20)
    today = date.today().isoformat()
    assert dated_path == tmp_path / f"{today}.md"
    assert dated_path.exists()
    latest_path = tmp_path / "latest.md"
    assert latest_path.exists()
    assert latest_path.read_text() == dated_path.read_text()


def test_render_digest_with_since_excludes_old_papers():
    conn = make_seeded_conn()
    # make_seeded_conn's paper was just inserted, so its first_seen_at is "now" --
    # backdate it to simulate an old paper outside the recency window.
    conn.execute(
        "UPDATE papers SET first_seen_at = ? WHERE arxiv_id = ?",
        ("2020-01-01T00:00:00+00:00", "2601.00001"),
    )
    conn.commit()
    text = digest.render_digest(conn, top_n=20, since="2025-01-01T00:00:00+00:00")
    assert "A Great Paper" not in text


def test_render_digest_with_since_includes_recent_papers():
    conn = make_seeded_conn()
    text = digest.render_digest(conn, top_n=20, since="2020-01-01T00:00:00+00:00")
    assert "A Great Paper" in text


def test_render_digest_without_since_is_unchanged():
    conn = make_seeded_conn()
    text = digest.render_digest(conn, top_n=20)
    assert "A Great Paper" in text


def test_select_digest_scores_orders_by_score_and_respects_top_n():
    conn = make_seeded_conn()
    db.insert_paper(conn, Paper(
        arxiv_id="2601.00002", title="Second Paper", authors="A", abstract="B",
        categories="cs.AI", published="2026-01-01T00:00:00Z", url="https://arxiv.org/abs/2601.00002",
    ))
    db.upsert_score(conn, Score(
        arxiv_id="2601.00002", similarity=0.1, feedback_adjustment=0.0, final_score=0.1,
        explanation="Weak match.", created_at="t",
    ))
    scores = digest.select_digest_scores(conn, top_n=1)
    assert [s.arxiv_id for s in scores] == ["2601.00001"]


def test_select_digest_scores_respects_since_filter():
    conn = make_seeded_conn()
    conn.execute(
        "UPDATE papers SET first_seen_at = ? WHERE arxiv_id = ?",
        ("2020-01-01T00:00:00+00:00", "2601.00001"),
    )
    conn.commit()
    scores = digest.select_digest_scores(conn, top_n=20, since="2025-01-01T00:00:00+00:00")
    assert scores == []


def test_render_digest_excludes_manual_source_even_if_recent():
    conn = make_seeded_conn()
    db.insert_paper(conn, Paper(
        arxiv_id="2601.00099", title="Manually Added Paper", authors="A", abstract="B",
        categories="cs.AI", published="2026-01-01T00:00:00Z", url="https://arxiv.org/abs/2601.00099",
    ), source="manual")
    db.upsert_score(conn, Score(
        arxiv_id="2601.00099", similarity=0.99, feedback_adjustment=0.0, final_score=0.99,
        explanation="Would otherwise rank first.", created_at="t",
    ))
    text = digest.render_digest(conn, top_n=20, since="2020-01-01T00:00:00+00:00")
    assert "Manually Added Paper" not in text
    assert "A Great Paper" in text
