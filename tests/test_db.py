from arxiv_curator import db
from arxiv_curator.models import Paper, Summary, Score, Feedback


def make_paper(arxiv_id="2601.00001"):
    return Paper(
        arxiv_id=arxiv_id, title="Title", authors="Author",
        abstract="Abstract text.", categories="cs.AI",
        published="2026-01-01T00:00:00Z", url=f"https://arxiv.org/abs/{arxiv_id}",
    )


def make_conn():
    conn = db.get_connection(":memory:")
    db.init_db(conn)
    return conn


def test_insert_and_get_paper():
    conn = make_conn()
    db.insert_paper(conn, make_paper())
    fetched = db.get_paper(conn, "2601.00001")
    assert fetched.title == "Title"


def test_insert_paper_is_idempotent():
    conn = make_conn()
    db.insert_paper(conn, make_paper())
    db.insert_paper(conn, make_paper())
    assert len(db.list_papers(conn)) == 1


def test_paper_exists():
    conn = make_conn()
    assert db.paper_exists(conn, "2601.00001") is False
    db.insert_paper(conn, make_paper())
    assert db.paper_exists(conn, "2601.00001") is True


def test_summary_round_trip_and_missing_query():
    conn = make_conn()
    db.insert_paper(conn, make_paper())
    assert db.papers_missing_summary(conn) == [db.get_paper(conn, "2601.00001")]
    db.insert_summary(conn, Summary(arxiv_id="2601.00001", text="A summary.", created_at="2026-01-01T00:00:00Z"))
    assert db.get_summary(conn, "2601.00001").text == "A summary."
    assert db.papers_missing_summary(conn) == []


def test_score_upsert_replaces_previous():
    conn = make_conn()
    db.insert_paper(conn, make_paper())
    db.upsert_score(conn, Score(
        arxiv_id="2601.00001", similarity=0.5, feedback_adjustment=0.0,
        final_score=0.5, explanation="first", created_at="2026-01-01T00:00:00Z",
    ))
    db.upsert_score(conn, Score(
        arxiv_id="2601.00001", similarity=0.6, feedback_adjustment=0.1,
        final_score=0.7, explanation="second", created_at="2026-01-02T00:00:00Z",
    ))
    scores = db.list_scores(conn)
    assert len(scores) == 1
    assert scores[0].explanation == "second"


def test_feedback_insert_and_list():
    conn = make_conn()
    db.insert_paper(conn, make_paper())
    db.insert_feedback(conn, Feedback(arxiv_id="2601.00001", created_at="2026-01-01T00:00:00Z", rating="up"))
    db.insert_feedback(conn, Feedback(arxiv_id="2601.00001", created_at="2026-01-02T00:00:00Z", pages_read=5, total_pages=10))
    items = db.list_feedback(conn)
    assert len(items) == 2
    assert items[0].rating == "up"
    assert items[1].pages_read == 5


def test_upsert_and_get_embeddings_round_trip():
    conn = make_conn()
    db.insert_paper(conn, make_paper("2601.00001"))
    db.upsert_embedding(conn, "2601.00001", [1.0, 0.5, -0.25])
    result = db.get_embeddings(conn, ["2601.00001"])
    assert list(result["2601.00001"]) == [1.0, 0.5, -0.25]


def test_get_embeddings_omits_uncached_ids():
    conn = make_conn()
    db.insert_paper(conn, make_paper("2601.00001"))
    result = db.get_embeddings(conn, ["2601.00001", "9999.00000"])
    assert "2601.00001" not in result
    assert "9999.00000" not in result


def test_upsert_embedding_replaces_previous():
    conn = make_conn()
    db.insert_paper(conn, make_paper("2601.00001"))
    db.upsert_embedding(conn, "2601.00001", [1.0, 0.0])
    db.upsert_embedding(conn, "2601.00001", [0.0, 1.0])
    result = db.get_embeddings(conn, ["2601.00001"])
    assert list(result["2601.00001"]) == [0.0, 1.0]


def test_list_papers_since_filters_by_first_seen_at():
    conn = make_conn()
    db.insert_paper(conn, make_paper("old1"))
    db.insert_paper(conn, make_paper("new1"))
    # Backdate "old1" to simulate it having been fetched before the cutoff window.
    conn.execute(
        "UPDATE papers SET first_seen_at = ? WHERE arxiv_id = ?",
        ("2020-01-01T00:00:00+00:00", "old1"),
    )
    conn.commit()
    cutoff = "2025-01-01T00:00:00+00:00"
    recent = db.list_papers_since(conn, cutoff)
    recent_ids = {p.arxiv_id for p in recent}
    assert "new1" in recent_ids
    assert "old1" not in recent_ids


def test_init_db_adds_source_column_to_pre_existing_papers_table():
    conn = db.get_connection(":memory:")
    conn.execute("""
        CREATE TABLE papers (
            arxiv_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            authors TEXT NOT NULL,
            abstract TEXT NOT NULL,
            categories TEXT NOT NULL,
            published TEXT NOT NULL,
            url TEXT NOT NULL,
            first_seen_at TEXT NOT NULL
        )
    """)
    conn.execute(
        "INSERT INTO papers (arxiv_id, title, authors, abstract, categories, published, url, first_seen_at) "
        "VALUES ('legacy1', 'T', 'A', 'B', 'cs.AI', 'p', 'u', 'f')"
    )
    conn.commit()

    db.init_db(conn)

    row = conn.execute("SELECT source FROM papers WHERE arxiv_id = 'legacy1'").fetchone()
    assert row["source"] == "fetch"


def test_insert_paper_defaults_to_fetch_source():
    conn = make_conn()
    db.insert_paper(conn, make_paper())
    row = conn.execute("SELECT source FROM papers WHERE arxiv_id = ?", ("2601.00001",)).fetchone()
    assert row["source"] == "fetch"


def test_insert_paper_accepts_manual_source():
    conn = make_conn()
    db.insert_paper(conn, make_paper(), source="manual")
    row = conn.execute("SELECT source FROM papers WHERE arxiv_id = ?", ("2601.00001",)).fetchone()
    assert row["source"] == "manual"
