from arxiv_curator import db, fetch
from arxiv_curator.models import Paper

SAMPLE_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2601.00001v1</id>
    <title>A Great Paper</title>
    <summary>This paper studies something interesting.</summary>
    <published>2026-01-01T00:00:00Z</published>
    <link href="http://arxiv.org/abs/2601.00001v1" rel="alternate"/>
    <author><name>Ada Author</name></author>
    <author><name>Bo Byte</name></author>
    <category term="cs.AI"/>
    <category term="cs.LG"/>
  </entry>
</feed>
"""


def test_build_query_url_includes_categories_and_max_results():
    url = fetch.build_query_url(["cs.AI", "cs.LG"], 50)
    assert "cat:cs.AI" in url
    assert "cat:cs.LG" in url
    assert "max_results=50" in url
    assert "sortBy=submittedDate" in url


def test_parse_feed_extracts_paper_fields():
    papers = fetch.parse_feed(SAMPLE_FEED)
    assert len(papers) == 1
    paper = papers[0]
    assert paper.arxiv_id == "2601.00001v1"
    assert paper.title == "A Great Paper"
    assert "Ada Author" in paper.authors
    assert "Bo Byte" in paper.authors
    assert "cs.AI" in paper.categories
    assert paper.url == "http://arxiv.org/abs/2601.00001v1"
    assert paper.pages is None


FEED_WITH_COMMENT = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2601.00001v1</id>
    <title>A Great Paper</title>
    <summary>This paper studies something interesting.</summary>
    <published>2026-01-01T00:00:00Z</published>
    <link href="http://arxiv.org/abs/2601.00001v1" rel="alternate"/>
    <author><name>Ada Author</name></author>
    <category term="cs.AI"/>
    <arxiv:comment>27 pages, 2 figures</arxiv:comment>
  </entry>
</feed>
"""


def test_parse_feed_sets_pages_from_arxiv_comment():
    papers = fetch.parse_feed(FEED_WITH_COMMENT)
    assert len(papers) == 1
    assert papers[0].pages == 27


EMPTY_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
</feed>
"""


class FakeResponse:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


def test_normalize_arxiv_id_passes_through_bare_id():
    assert fetch.normalize_arxiv_id("2601.00001") == "2601.00001"


def test_normalize_arxiv_id_extracts_id_from_url():
    assert fetch.normalize_arxiv_id("https://arxiv.org/abs/2601.00001") == "2601.00001"


def test_normalize_arxiv_id_strips_trailing_slash():
    assert fetch.normalize_arxiv_id("https://arxiv.org/abs/2601.00001/") == "2601.00001"


def test_build_id_query_url_includes_id():
    url = fetch.build_id_query_url("2601.00001")
    assert "id_list=2601.00001" in url


def test_build_ids_query_url_joins_multiple_ids():
    url = fetch.build_ids_query_url(["2601.00001", "2601.00002", "2601.00003"])
    assert "id_list=2601.00001,2601.00002,2601.00003" in url


def test_fetch_paper_by_id_returns_paper_when_found(monkeypatch):
    def fake_get(url, timeout):
        return FakeResponse(SAMPLE_FEED)

    monkeypatch.setattr(fetch.requests, "get", fake_get)
    paper = fetch.fetch_paper_by_id("2601.00001")
    assert paper is not None
    assert paper.arxiv_id == "2601.00001v1"
    assert paper.title == "A Great Paper"


def test_fetch_paper_by_id_returns_none_when_not_found(monkeypatch):
    def fake_get(url, timeout):
        return FakeResponse(EMPTY_FEED)

    monkeypatch.setattr(fetch.requests, "get", fake_get)
    assert fetch.fetch_paper_by_id("9999.99999") is None


def test_fetch_and_store_dedups_existing_papers(monkeypatch):
    conn = db.get_connection(":memory:")
    db.init_db(conn)
    existing = Paper(
        arxiv_id="2601.00001v1", title="Old title", authors="X", abstract="Y",
        categories="cs.AI", published="2026-01-01T00:00:00Z", url="http://arxiv.org/abs/2601.00001v1",
    )
    db.insert_paper(conn, existing)

    def fake_fetch_papers(categories, max_results):
        return fetch.parse_feed(SAMPLE_FEED)

    monkeypatch.setattr(fetch, "fetch_papers", fake_fetch_papers)
    new_count = fetch.fetch_and_store(conn, ["cs.AI"], 10)
    assert new_count == 0
    assert len(db.list_papers(conn)) == 1


def test_fetch_and_store_inserts_with_fetch_source(monkeypatch):
    conn = db.get_connection(":memory:")
    db.init_db(conn)

    def fake_fetch_papers(categories, max_results):
        return fetch.parse_feed(SAMPLE_FEED)

    monkeypatch.setattr(fetch, "fetch_papers", fake_fetch_papers)
    fetch.fetch_and_store(conn, ["cs.AI"], 10)

    row = conn.execute(
        "SELECT source FROM papers WHERE arxiv_id = ?", ("2601.00001v1",)
    ).fetchone()
    assert row["source"] == "fetch"


def test_extract_page_count_finds_leading_page_count():
    assert fetch.extract_page_count("27 pages, 2 figures, 1 table") == 27


def test_extract_page_count_handles_singular_page():
    assert fetch.extract_page_count("1 page") == 1


def test_extract_page_count_returns_none_without_page_mention():
    assert fetch.extract_page_count("Code and demo: https://example.com") is None


def test_extract_page_count_returns_none_for_none_input():
    assert fetch.extract_page_count(None) is None


def test_extract_page_count_is_case_insensitive():
    assert fetch.extract_page_count("8 Pages, 4 Figures") == 8


TWO_ENTRY_FEED_WITH_COMMENTS = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2601.00001v1</id>
    <title>First Paper</title>
    <summary>Summary one.</summary>
    <published>2026-01-01T00:00:00Z</published>
    <link href="http://arxiv.org/abs/2601.00001v1" rel="alternate"/>
    <author><name>Ada Author</name></author>
    <category term="cs.AI"/>
    <arxiv:comment>12 pages, 3 figures</arxiv:comment>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2601.00002v1</id>
    <title>Second Paper</title>
    <summary>Summary two.</summary>
    <published>2026-01-01T00:00:00Z</published>
    <link href="http://arxiv.org/abs/2601.00002v1" rel="alternate"/>
    <author><name>Bo Byte</name></author>
    <category term="cs.AI"/>
  </entry>
</feed>
"""


def test_backfill_pages_updates_only_papers_with_parseable_comment(monkeypatch):
    conn = db.get_connection(":memory:")
    db.init_db(conn)
    db.insert_paper(conn, Paper(
        arxiv_id="2601.00001v1", title="First Paper", authors="Ada Author",
        abstract="Summary one.", categories="cs.AI",
        published="2026-01-01T00:00:00Z", url="http://arxiv.org/abs/2601.00001v1",
    ))
    db.insert_paper(conn, Paper(
        arxiv_id="2601.00002v1", title="Second Paper", authors="Bo Byte",
        abstract="Summary two.", categories="cs.AI",
        published="2026-01-01T00:00:00Z", url="http://arxiv.org/abs/2601.00002v1",
    ))

    def fake_get(url, timeout):
        return FakeResponse(TWO_ENTRY_FEED_WITH_COMMENTS)

    monkeypatch.setattr(fetch.requests, "get", fake_get)
    updated_count = fetch.backfill_pages(conn)

    assert updated_count == 1
    assert db.get_paper(conn, "2601.00001v1").pages == 12
    assert db.get_paper(conn, "2601.00002v1").pages is None


def test_backfill_pages_returns_zero_when_nothing_missing():
    conn = db.get_connection(":memory:")
    db.init_db(conn)
    assert fetch.backfill_pages(conn) == 0


def test_backfill_pages_chunks_requests_by_fifty(monkeypatch):
    conn = db.get_connection(":memory:")
    db.init_db(conn)
    for i in range(120):
        arxiv_id = f"2601.{i:05d}"
        db.insert_paper(conn, Paper(
            arxiv_id=arxiv_id, title="T", authors="A", abstract="B",
            categories="cs.AI", published="2026-01-01T00:00:00Z",
            url=f"http://arxiv.org/abs/{arxiv_id}",
        ))

    calls = []

    def fake_get(url, timeout):
        calls.append(url)
        return FakeResponse(EMPTY_FEED)

    monkeypatch.setattr(fetch.requests, "get", fake_get)
    fetch.backfill_pages(conn)

    assert len(calls) == 3  # 120 papers / 50 per chunk = 3 requests
