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
