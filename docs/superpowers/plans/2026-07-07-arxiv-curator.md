# arXiv Curator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python backend/CLI that fetches newest arXiv papers, summarizes them via Gemini, ranks them against a personal interest profile with Rocchio-style feedback adjustment, records feedback (rating + self-reported read depth), explains "why this matches," runs a ranking-quality eval, and publishes committed markdown digests viewable on GitHub.

**Architecture:** SQLite-backed Typer CLI. Pure math (cosine similarity, Rocchio centroids, ranking metrics) lives in small testable functions with no I/O; orchestration functions wire those to the db and Gemini client. Gemini is the only LLM provider (chat + embeddings), behind protocols general enough to add a second provider later without touching callers.

**Tech Stack:** Python 3.11+, Typer, SQLite (stdlib `sqlite3`), `requests` + `feedparser` for arXiv, `google-genai` for Gemini (chat + embeddings), `pyyaml`, `numpy`, `python-dotenv`, `pytest`.

## Global Constraints

- Python >= 3.11.
- Package layout: source under `src/arxiv_curator/`, installed editable (`pip install -e ".[dev]"`).
- No local/offline LLM fallback and no second LLM provider in v1 — Gemini only, via `GEMINI_API_KEY`.
- Embeddings use Gemini's embeddings API (`text-embedding-004`), not a local model.
- `data/arxiv_curator.db` stays gitignored (holds personal feedback); never commit it.
- `digests/*.md` are committed — this is what makes the project "viewable on GitHub."
- No automated test hits the real Gemini API — provider/embedding wrappers are tested against injected fakes; the actual API is exercised only via manual verification once a key is available.
- Every module with pure logic (`rank.py` scoring functions, `eval.py` metrics) must be unit-testable with hand-computed fixed vectors, no network.

---

## File Structure

```
arxiv-curator/
  README.md
  pyproject.toml
  .env.example
  .gitignore
  interests.yaml
  src/arxiv_curator/
    __init__.py
    models.py
    db.py
    interests.py
    fetch.py
    rank.py
    feedback.py
    digest.py
    eval.py
    cli.py
    llm/
      __init__.py
      base.py
      factory.py
      embeddings.py
      gemini_provider.py
  tests/
    __init__.py
    test_db.py
    test_fetch.py
    test_rank.py
    test_eval.py
  digests/
    .gitkeep
  .github/workflows/
    ci.yml
    daily-digest.yml
```

`interests.py` is a small addition beyond the original spec sketch (which folded interest-profile loading into `config.py`/`rank.py`) — it's split out because parsing/validating `interests.yaml` and turning it into embeddable text is a distinct responsibility from ranking math.

---

### Task 1: Project scaffolding + models.py

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `src/arxiv_curator/__init__.py` (empty)
- Create: `src/arxiv_curator/models.py`
- Create: `tests/__init__.py` (empty)
- Test: `tests/test_models.py`

**Interfaces:**
- Produces: `Paper(arxiv_id, title, authors, abstract, categories, published, url)`, `Summary(arxiv_id, text, created_at)`, `Score(arxiv_id, similarity, feedback_adjustment, final_score, explanation, created_at)`, `Feedback(arxiv_id, rating, pages_read, total_pages, note, created_at)` — all plain `@dataclass`, all fields on `Feedback` except `arxiv_id`/`created_at` default to `None`.

- [ ] **Step 1: Create project files**

`pyproject.toml`:
```toml
[project]
name = "arxiv-curator"
version = "0.1.0"
description = "Personal backend/CLI for fetching, summarizing, and ranking arXiv papers against your interests"
requires-python = ">=3.11"
dependencies = [
    "typer>=0.12",
    "requests>=2.31",
    "feedparser>=6.0",
    "pyyaml>=6.0",
    "numpy>=1.26",
    "python-dotenv>=1.0",
    "google-genai>=0.3",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[project.scripts]
arxiv-curator = "arxiv_curator.cli:app"

[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
```

`.gitignore`:
```
data/
.env
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
.venv/
```

`.env.example`:
```
GEMINI_API_KEY=your-gemini-api-key-here
```

- [ ] **Step 2: Write the failing test**

`tests/test_models.py`:
```python
from arxiv_curator.models import Paper, Summary, Score, Feedback


def test_paper_fields():
    paper = Paper(
        arxiv_id="2601.00001",
        title="A Paper",
        authors="A. Author",
        abstract="An abstract.",
        categories="cs.AI",
        published="2026-01-01T00:00:00Z",
        url="https://arxiv.org/abs/2601.00001",
    )
    assert paper.arxiv_id == "2601.00001"


def test_feedback_defaults_to_none():
    feedback = Feedback(arxiv_id="2601.00001", created_at="2026-01-01T00:00:00Z")
    assert feedback.rating is None
    assert feedback.pages_read is None
    assert feedback.total_pages is None
    assert feedback.note is None


def test_feedback_with_rating_and_read_depth():
    feedback = Feedback(
        arxiv_id="2601.00001",
        created_at="2026-01-01T00:00:00Z",
        rating="up",
        pages_read=5,
        total_pages=12,
        note="good background section",
    )
    assert feedback.rating == "up"
    assert feedback.pages_read == 5
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd arxiv-curator && python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]" && pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'arxiv_curator.models'` (or similar import error), since `src/arxiv_curator/models.py` doesn't exist yet — but `pip install -e` will fail first if the package dir is missing, so create an empty `src/arxiv_curator/__init__.py` before running.

- [ ] **Step 4: Write minimal implementation**

`src/arxiv_curator/models.py`:
```python
from dataclasses import dataclass
from typing import Optional


@dataclass
class Paper:
    arxiv_id: str
    title: str
    authors: str
    abstract: str
    categories: str
    published: str
    url: str


@dataclass
class Summary:
    arxiv_id: str
    text: str
    created_at: str


@dataclass
class Score:
    arxiv_id: str
    similarity: float
    feedback_adjustment: float
    final_score: float
    explanation: str
    created_at: str


@dataclass
class Feedback:
    arxiv_id: str
    created_at: str
    rating: Optional[str] = None
    pages_read: Optional[int] = None
    total_pages: Optional[int] = None
    note: Optional[str] = None
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_models.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore .env.example src/arxiv_curator/__init__.py src/arxiv_curator/models.py tests/__init__.py tests/test_models.py
git commit -m "Scaffold project and add data models"
```

---

### Task 2: db.py — SQLite schema + CRUD

**Files:**
- Create: `src/arxiv_curator/db.py`
- Test: `tests/test_db.py`

**Interfaces:**
- Consumes: `Paper`, `Summary`, `Score`, `Feedback` from `arxiv_curator.models` (Task 1).
- Produces: `get_connection(db_path) -> sqlite3.Connection`, `init_db(conn)`, `insert_paper(conn, paper)`, `paper_exists(conn, arxiv_id) -> bool`, `get_paper(conn, arxiv_id) -> Optional[Paper]`, `list_papers(conn) -> list[Paper]`, `insert_summary(conn, summary)`, `get_summary(conn, arxiv_id) -> Optional[Summary]`, `papers_missing_summary(conn) -> list[Paper]`, `upsert_score(conn, score)`, `get_score(conn, arxiv_id) -> Optional[Score]`, `list_scores(conn) -> list[Score]`, `insert_feedback(conn, feedback)`, `list_feedback(conn) -> list[Feedback]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_db.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_db.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'arxiv_curator.db'`

- [ ] **Step 3: Write implementation**

`src/arxiv_curator/db.py`:
```python
import sqlite3
from typing import Optional

from arxiv_curator.models import Paper, Summary, Score, Feedback

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    arxiv_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    authors TEXT NOT NULL,
    abstract TEXT NOT NULL,
    categories TEXT NOT NULL,
    published TEXT NOT NULL,
    url TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS summaries (
    arxiv_id TEXT PRIMARY KEY REFERENCES papers(arxiv_id),
    text TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scores (
    arxiv_id TEXT PRIMARY KEY REFERENCES papers(arxiv_id),
    similarity REAL NOT NULL,
    feedback_adjustment REAL NOT NULL,
    final_score REAL NOT NULL,
    explanation TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    arxiv_id TEXT NOT NULL REFERENCES papers(arxiv_id),
    rating TEXT,
    pages_read INTEGER,
    total_pages INTEGER,
    note TEXT,
    created_at TEXT NOT NULL
);
"""


def get_connection(db_path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def _row_to_paper(row) -> Paper:
    return Paper(
        arxiv_id=row["arxiv_id"], title=row["title"], authors=row["authors"],
        abstract=row["abstract"], categories=row["categories"],
        published=row["published"], url=row["url"],
    )


def _row_to_score(row) -> Score:
    return Score(
        arxiv_id=row["arxiv_id"], similarity=row["similarity"],
        feedback_adjustment=row["feedback_adjustment"], final_score=row["final_score"],
        explanation=row["explanation"], created_at=row["created_at"],
    )


def _row_to_feedback(row) -> Feedback:
    return Feedback(
        arxiv_id=row["arxiv_id"], created_at=row["created_at"], rating=row["rating"],
        pages_read=row["pages_read"], total_pages=row["total_pages"], note=row["note"],
    )


def insert_paper(conn, paper: Paper) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO papers (arxiv_id, title, authors, abstract, categories, published, url) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (paper.arxiv_id, paper.title, paper.authors, paper.abstract, paper.categories, paper.published, paper.url),
    )
    conn.commit()


def paper_exists(conn, arxiv_id: str) -> bool:
    row = conn.execute("SELECT 1 FROM papers WHERE arxiv_id = ?", (arxiv_id,)).fetchone()
    return row is not None


def get_paper(conn, arxiv_id: str) -> Optional[Paper]:
    row = conn.execute("SELECT * FROM papers WHERE arxiv_id = ?", (arxiv_id,)).fetchone()
    return _row_to_paper(row) if row else None


def list_papers(conn) -> list[Paper]:
    rows = conn.execute("SELECT * FROM papers").fetchall()
    return [_row_to_paper(row) for row in rows]


def insert_summary(conn, summary: Summary) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO summaries (arxiv_id, text, created_at) VALUES (?, ?, ?)",
        (summary.arxiv_id, summary.text, summary.created_at),
    )
    conn.commit()


def get_summary(conn, arxiv_id: str) -> Optional[Summary]:
    row = conn.execute("SELECT * FROM summaries WHERE arxiv_id = ?", (arxiv_id,)).fetchone()
    return Summary(arxiv_id=row["arxiv_id"], text=row["text"], created_at=row["created_at"]) if row else None


def papers_missing_summary(conn) -> list[Paper]:
    rows = conn.execute(
        "SELECT p.* FROM papers p LEFT JOIN summaries s ON p.arxiv_id = s.arxiv_id WHERE s.arxiv_id IS NULL"
    ).fetchall()
    return [_row_to_paper(row) for row in rows]


def upsert_score(conn, score: Score) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO scores "
        "(arxiv_id, similarity, feedback_adjustment, final_score, explanation, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (score.arxiv_id, score.similarity, score.feedback_adjustment, score.final_score,
         score.explanation, score.created_at),
    )
    conn.commit()


def get_score(conn, arxiv_id: str) -> Optional[Score]:
    row = conn.execute("SELECT * FROM scores WHERE arxiv_id = ?", (arxiv_id,)).fetchone()
    return _row_to_score(row) if row else None


def list_scores(conn) -> list[Score]:
    rows = conn.execute("SELECT * FROM scores").fetchall()
    return [_row_to_score(row) for row in rows]


def insert_feedback(conn, feedback: Feedback) -> None:
    conn.execute(
        "INSERT INTO feedback (arxiv_id, rating, pages_read, total_pages, note, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (feedback.arxiv_id, feedback.rating, feedback.pages_read, feedback.total_pages,
         feedback.note, feedback.created_at),
    )
    conn.commit()


def list_feedback(conn) -> list[Feedback]:
    rows = conn.execute("SELECT * FROM feedback ORDER BY id").fetchall()
    return [_row_to_feedback(row) for row in rows]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_db.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/arxiv_curator/db.py tests/test_db.py
git commit -m "Add SQLite schema and CRUD layer"
```

---

### Task 3: interests.py — interest profile loading

**Files:**
- Create: `src/arxiv_curator/interests.py`
- Test: `tests/test_interests.py`

**Interfaces:**
- Produces: `InterestProfile(summary, topics, keywords, liked_examples)` dataclass, `load_interest_profile(path: Path) -> InterestProfile`, `profile_to_text(profile: InterestProfile) -> str`.

- [ ] **Step 1: Write the failing tests**

`tests/test_interests.py`:
```python
import pytest
from arxiv_curator.interests import load_interest_profile, profile_to_text


def test_load_full_profile(tmp_path):
    path = tmp_path / "interests.yaml"
    path.write_text(
        "summary: I like retrieval-augmented generation and agent evaluation.\n"
        "topics:\n  - RAG\n  - agents\n"
        "keywords:\n  - retrieval\n  - evaluation\n"
        "liked_examples:\n  - \"Self-RAG\"\n"
    )
    profile = load_interest_profile(path)
    assert "retrieval-augmented" in profile.summary
    assert profile.topics == ["RAG", "agents"]
    assert profile.keywords == ["retrieval", "evaluation"]
    assert profile.liked_examples == ["Self-RAG"]


def test_load_minimal_profile_defaults_lists(tmp_path):
    path = tmp_path / "interests.yaml"
    path.write_text("summary: Just a summary.\n")
    profile = load_interest_profile(path)
    assert profile.topics == []
    assert profile.keywords == []
    assert profile.liked_examples == []


def test_missing_summary_raises(tmp_path):
    path = tmp_path / "interests.yaml"
    path.write_text("topics:\n  - RAG\n")
    with pytest.raises(ValueError, match="summary"):
        load_interest_profile(path)


def test_profile_to_text_includes_all_sections():
    from arxiv_curator.interests import InterestProfile
    profile = InterestProfile(
        summary="I like RAG.", topics=["RAG"], keywords=["retrieval"], liked_examples=["Self-RAG"],
    )
    text = profile_to_text(profile)
    assert "I like RAG." in text
    assert "RAG" in text
    assert "retrieval" in text
    assert "Self-RAG" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_interests.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'arxiv_curator.interests'`

- [ ] **Step 3: Write implementation**

`src/arxiv_curator/interests.py`:
```python
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class InterestProfile:
    summary: str
    topics: list[str]
    keywords: list[str]
    liked_examples: list[str]


def load_interest_profile(path: Path) -> InterestProfile:
    data = yaml.safe_load(Path(path).read_text()) or {}
    if "summary" not in data or not data["summary"]:
        raise ValueError("interests.yaml must include a non-empty 'summary' field")
    return InterestProfile(
        summary=data["summary"],
        topics=data.get("topics") or [],
        keywords=data.get("keywords") or [],
        liked_examples=data.get("liked_examples") or [],
    )


def profile_to_text(profile: InterestProfile) -> str:
    parts = [profile.summary]
    if profile.topics:
        parts.append("Topics: " + ", ".join(profile.topics))
    if profile.keywords:
        parts.append("Keywords: " + ", ".join(profile.keywords))
    if profile.liked_examples:
        parts.append("Examples of papers I like: " + "; ".join(profile.liked_examples))
    return "\n".join(parts)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_interests.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/arxiv_curator/interests.py tests/test_interests.py
git commit -m "Add interest profile loading"
```

---

### Task 4: fetch.py — arXiv client + dedup

**Files:**
- Create: `src/arxiv_curator/fetch.py`
- Test: `tests/test_fetch.py`

**Interfaces:**
- Consumes: `db.paper_exists`, `db.insert_paper` (Task 2); `Paper` (Task 1).
- Produces: `build_query_url(categories: list[str], max_results: int) -> str`, `parse_feed(raw_text: str) -> list[Paper]`, `fetch_papers(categories: list[str], max_results: int) -> list[Paper]`, `fetch_and_store(conn, categories: list[str], max_results: int) -> int`.

- [ ] **Step 1: Write the failing tests**

`tests/test_fetch.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_fetch.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'arxiv_curator.fetch'`

- [ ] **Step 3: Write implementation**

`src/arxiv_curator/fetch.py`:
```python
import feedparser
import requests

from arxiv_curator import db
from arxiv_curator.models import Paper

ARXIV_API_URL = "http://export.arxiv.org/api/query"


def build_query_url(categories: list[str], max_results: int) -> str:
    cat_query = "+OR+".join(f"cat:{c}" for c in categories)
    return (
        f"{ARXIV_API_URL}?search_query={cat_query}"
        f"&sortBy=submittedDate&sortOrder=descending&max_results={max_results}"
    )


def parse_feed(raw_text: str) -> list[Paper]:
    feed = feedparser.parse(raw_text)
    papers = []
    for entry in feed.entries:
        arxiv_id = entry.id.split("/abs/")[-1]
        authors = ", ".join(a.name for a in entry.get("authors", []))
        categories = ", ".join(t.term for t in entry.get("tags", []))
        papers.append(Paper(
            arxiv_id=arxiv_id,
            title=entry.title.replace("\n", " ").strip(),
            authors=authors,
            abstract=entry.summary.replace("\n", " ").strip(),
            categories=categories,
            published=entry.published,
            url=entry.link,
        ))
    return papers


def fetch_papers(categories: list[str], max_results: int) -> list[Paper]:
    url = build_query_url(categories, max_results)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return parse_feed(response.text)


def fetch_and_store(conn, categories: list[str], max_results: int) -> int:
    papers = fetch_papers(categories, max_results)
    new_count = 0
    for paper in papers:
        if not db.paper_exists(conn, paper.arxiv_id):
            db.insert_paper(conn, paper)
            new_count += 1
    return new_count
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_fetch.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/arxiv_curator/fetch.py tests/test_fetch.py
git commit -m "Add arXiv fetch client with dedup"
```

---

### Task 5: llm/base.py + llm/factory.py + llm/embeddings.py

**Files:**
- Create: `src/arxiv_curator/llm/__init__.py` (empty)
- Create: `src/arxiv_curator/llm/base.py`
- Create: `src/arxiv_curator/llm/factory.py`
- Create: `src/arxiv_curator/llm/embeddings.py`
- Test: `tests/test_embeddings.py`

**Interfaces:**
- Produces: `Summarizer` / `Explainer` Protocols (`base.py`); `get_client() -> genai.Client` (`factory.py`, raises `RuntimeError` if `GEMINI_API_KEY` unset); `embed_texts(texts: list[str], client) -> np.ndarray` and `cosine_similarity(a: np.ndarray, b: np.ndarray) -> float` (`embeddings.py`).

- [ ] **Step 1: Write the failing tests**

`tests/test_embeddings.py`:
```python
from types import SimpleNamespace

import numpy as np
import pytest

from arxiv_curator.llm.embeddings import embed_texts, cosine_similarity
from arxiv_curator.llm import factory


class FakeModels:
    def __init__(self, vectors):
        self._vectors = vectors

    def embed_content(self, model, contents):
        return SimpleNamespace(embeddings=[SimpleNamespace(values=v) for v in self._vectors])


class FakeClient:
    def __init__(self, vectors):
        self.models = FakeModels(vectors)


def test_embed_texts_returns_matrix_matching_input_order():
    client = FakeClient(vectors=[[1.0, 0.0], [0.0, 1.0]])
    result = embed_texts(["a", "b"], client)
    assert result.shape == (2, 2)
    assert list(result[0]) == [1.0, 0.0]
    assert list(result[1]) == [0.0, 1.0]


def test_embed_texts_empty_input_returns_empty_array():
    client = FakeClient(vectors=[])
    result = embed_texts([], client)
    assert result.shape[0] == 0


def test_cosine_similarity_identical_vectors_is_one():
    a = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(a, a) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector_is_zero_not_nan():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 0.0])
    assert cosine_similarity(a, b) == 0.0


def test_get_client_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        factory.get_client()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_embeddings.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'arxiv_curator.llm'`

- [ ] **Step 3: Write implementation**

`src/arxiv_curator/llm/base.py`:
```python
from typing import Protocol

from arxiv_curator.models import Paper


class Summarizer(Protocol):
    def summarize(self, paper: Paper) -> str: ...


class Explainer(Protocol):
    def explain(self, paper: Paper, interest_profile_text: str, signals: dict) -> str: ...
```

`src/arxiv_curator/llm/factory.py`:
```python
import os

from google import genai


def get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to your .env file or environment."
        )
    return genai.Client(api_key=api_key)
```

`src/arxiv_curator/llm/embeddings.py`:
```python
import numpy as np

EMBEDDING_MODEL = "text-embedding-004"


def embed_texts(texts: list[str], client) -> np.ndarray:
    if not texts:
        return np.empty((0, 0))
    result = client.models.embed_content(model=EMBEDDING_MODEL, contents=texts)
    return np.array([embedding.values for embedding in result.embeddings])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_embeddings.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/arxiv_curator/llm/__init__.py src/arxiv_curator/llm/base.py src/arxiv_curator/llm/factory.py src/arxiv_curator/llm/embeddings.py tests/test_embeddings.py
git commit -m "Add LLM protocols, Gemini client factory, and embeddings wrapper"
```

---

### Task 6: rank.py — pure Rocchio scoring functions

**Files:**
- Create: `src/arxiv_curator/rank.py` (pure functions only for this task)
- Test: `tests/test_rank.py`

**Interfaces:**
- Consumes: `cosine_similarity` (Task 5); `Feedback` (Task 1).
- Produces: `feedback_weight(feedback: Feedback) -> float`, `compute_centroids(feedback_items: list[Feedback], vectors_by_id: dict[str, np.ndarray]) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]`, `score_paper(paper_vector, interest_vector, mean_liked, mean_disliked, alpha=0.3, beta=0.3) -> tuple[float, float, float]`, `overlapping_keywords(abstract: str, keywords: list[str]) -> list[str]`, `most_similar_liked(paper_vector, liked_vectors_by_id: dict[str, np.ndarray]) -> Optional[str]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_rank.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_rank.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'arxiv_curator.rank'`

- [ ] **Step 3: Write implementation**

`src/arxiv_curator/rank.py`:
```python
from typing import Optional

import numpy as np

from arxiv_curator.llm.embeddings import cosine_similarity
from arxiv_curator.models import Feedback

DEFAULT_ALPHA = 0.3
DEFAULT_BETA = 0.3


def feedback_weight(feedback: Feedback) -> float:
    if feedback.pages_read is not None and feedback.total_pages:
        return 0.5 + 0.5 * (feedback.pages_read / feedback.total_pages)
    return 1.0


def compute_centroids(
    feedback_items: list[Feedback], vectors_by_id: dict[str, np.ndarray]
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    liked_vectors = []
    disliked_vectors = []
    for fb in feedback_items:
        vec = vectors_by_id.get(fb.arxiv_id)
        if vec is None:
            continue
        weight = feedback_weight(fb)
        if fb.rating == "up":
            liked_vectors.append(vec * weight)
        elif fb.rating == "down":
            disliked_vectors.append(vec * weight)
    mean_liked = np.mean(liked_vectors, axis=0) if liked_vectors else None
    mean_disliked = np.mean(disliked_vectors, axis=0) if disliked_vectors else None
    return mean_liked, mean_disliked


def score_paper(
    paper_vector: np.ndarray,
    interest_vector: np.ndarray,
    mean_liked: Optional[np.ndarray],
    mean_disliked: Optional[np.ndarray],
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
) -> tuple[float, float, float]:
    similarity = cosine_similarity(paper_vector, interest_vector)
    adjustment = 0.0
    if mean_liked is not None:
        adjustment += alpha * cosine_similarity(paper_vector, mean_liked)
    if mean_disliked is not None:
        adjustment -= beta * cosine_similarity(paper_vector, mean_disliked)
    final = similarity + adjustment
    return similarity, adjustment, final


def overlapping_keywords(abstract: str, keywords: list[str]) -> list[str]:
    abstract_lower = abstract.lower()
    return [kw for kw in keywords if kw.lower() in abstract_lower]


def most_similar_liked(
    paper_vector: np.ndarray, liked_vectors_by_id: dict[str, np.ndarray]
) -> Optional[str]:
    best_id = None
    best_score = -1.0
    for arxiv_id, vec in liked_vectors_by_id.items():
        sim = cosine_similarity(paper_vector, vec)
        if sim > best_score:
            best_id, best_score = arxiv_id, sim
    return best_id
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_rank.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add src/arxiv_curator/rank.py tests/test_rank.py
git commit -m "Add Rocchio-style ranking math"
```

---

### Task 7: llm/gemini_provider.py

**Files:**
- Create: `src/arxiv_curator/llm/gemini_provider.py`
- Test: `tests/test_gemini_provider.py`

**Interfaces:**
- Consumes: `Paper` (Task 1).
- Produces: `build_summarize_prompt(paper: Paper) -> str`, `build_explain_prompt(paper: Paper, interest_profile_text: str, signals: dict) -> str`, `GeminiProvider(client, model: str = "gemini-2.5-flash")` implementing `summarize(paper) -> str` and `explain(paper, interest_profile_text, signals) -> str`.

- [ ] **Step 1: Write the failing tests**

`tests/test_gemini_provider.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_gemini_provider.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'arxiv_curator.llm.gemini_provider'`

- [ ] **Step 3: Write implementation**

`src/arxiv_curator/llm/gemini_provider.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_gemini_provider.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/arxiv_curator/llm/gemini_provider.py tests/test_gemini_provider.py
git commit -m "Add Gemini-backed summarizer and explainer"
```

---

### Task 8: rank.py orchestration — rank_papers

**Files:**
- Modify: `src/arxiv_curator/rank.py` (append orchestration function)
- Modify: `tests/test_rank.py` (append orchestration test)

**Interfaces:**
- Consumes: `db.list_papers`, `db.list_feedback`, `db.upsert_score` (Task 2); `load_interest_profile`, `profile_to_text` (Task 3); `embed_texts` (Task 5); `Explainer` protocol (Task 5); `compute_centroids`, `score_paper`, `overlapping_keywords`, `most_similar_liked` (Task 6).
- Produces: `rank_papers(conn, interests_path: Path, provider, client) -> list[Score]`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_rank.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rank.py::test_rank_papers_scores_relevant_paper_higher -v`
Expected: FAIL with `AttributeError: module 'arxiv_curator.rank' has no attribute 'rank_papers'`

- [ ] **Step 3: Write implementation**

Append to `src/arxiv_curator/rank.py`:
```python
from datetime import datetime, timezone

from arxiv_curator import db
from arxiv_curator.interests import load_interest_profile, profile_to_text
from arxiv_curator.llm.embeddings import embed_texts
from arxiv_curator.models import Score


def rank_papers(conn, interests_path, provider, client) -> list[Score]:
    profile = load_interest_profile(interests_path)
    profile_text = profile_to_text(profile)
    interest_vector = embed_texts([profile_text], client)[0]

    papers = db.list_papers(conn)
    if not papers:
        return []
    abstracts = [p.abstract for p in papers]
    paper_vectors = embed_texts(abstracts, client)
    vectors_by_id = {p.arxiv_id: vec for p, vec in zip(papers, paper_vectors)}

    feedback_items = db.list_feedback(conn)
    liked_ids = {fb.arxiv_id for fb in feedback_items if fb.rating == "up"}
    liked_vectors_by_id = {aid: vectors_by_id[aid] for aid in liked_ids if aid in vectors_by_id}
    mean_liked, mean_disliked = compute_centroids(feedback_items, vectors_by_id)

    results = []
    for paper in papers:
        vec = vectors_by_id[paper.arxiv_id]
        similarity, adjustment, final = score_paper(vec, interest_vector, mean_liked, mean_disliked)
        keywords = overlapping_keywords(paper.abstract, profile.keywords)
        closest = most_similar_liked(vec, liked_vectors_by_id)
        signals = {"overlapping_keywords": keywords, "most_similar_liked": closest}
        explanation = provider.explain(paper, profile_text, signals)
        score = Score(
            arxiv_id=paper.arxiv_id, similarity=similarity, feedback_adjustment=adjustment,
            final_score=final, explanation=explanation,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        db.upsert_score(conn, score)
        results.append(score)
    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_rank.py -v`
Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
git add src/arxiv_curator/rank.py tests/test_rank.py
git commit -m "Add rank_papers orchestration wiring embeddings, feedback, and the explainer"
```

---

### Task 9: feedback.py

**Files:**
- Create: `src/arxiv_curator/feedback.py`
- Test: `tests/test_feedback.py`

**Interfaces:**
- Consumes: `db.insert_feedback`, `db.list_feedback`, `db.paper_exists` (Task 2); `Feedback` (Task 1).
- Produces: `record_feedback(conn, arxiv_id, rating=None, note=None, pages_read=None, total_pages=None) -> None` (raises `ValueError` on invalid rating, on unknown `arxiv_id`, or if `pages_read > total_pages`).

- [ ] **Step 1: Write the failing tests**

`tests/test_feedback.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_feedback.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'arxiv_curator.feedback'`

- [ ] **Step 3: Write implementation**

`src/arxiv_curator/feedback.py`:
```python
from datetime import datetime, timezone
from typing import Optional

from arxiv_curator import db
from arxiv_curator.models import Feedback

VALID_RATINGS = {"up", "down"}


def record_feedback(
    conn,
    arxiv_id: str,
    rating: Optional[str] = None,
    note: Optional[str] = None,
    pages_read: Optional[int] = None,
    total_pages: Optional[int] = None,
) -> None:
    if not db.paper_exists(conn, arxiv_id):
        raise ValueError(f"No such paper: {arxiv_id}")
    if rating is not None and rating not in VALID_RATINGS:
        raise ValueError(f"rating must be one of {VALID_RATINGS}, got {rating!r}")
    if pages_read is not None and total_pages is not None and pages_read > total_pages:
        raise ValueError("pages_read cannot exceed total_pages")

    db.insert_feedback(conn, Feedback(
        arxiv_id=arxiv_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        rating=rating,
        pages_read=pages_read,
        total_pages=total_pages,
        note=note,
    ))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_feedback.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/arxiv_curator/feedback.py tests/test_feedback.py
git commit -m "Add feedback recording with validation"
```

---

### Task 10: digest.py

**Files:**
- Create: `src/arxiv_curator/digest.py`
- Test: `tests/test_digest.py`

**Interfaces:**
- Consumes: `db.list_scores`, `db.get_paper`, `db.get_summary` (Task 2).
- Produces: `render_digest(conn, top_n: int = 20) -> str`, `write_digest(conn, out_dir: Path, top_n: int = 20) -> Path` (writes both `out_dir/YYYY-MM-DD.md` and `out_dir/latest.md`, returns the dated path).

- [ ] **Step 1: Write the failing tests**

`tests/test_digest.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_digest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'arxiv_curator.digest'`

- [ ] **Step 3: Write implementation**

`src/arxiv_curator/digest.py`:
```python
from datetime import date
from pathlib import Path

from arxiv_curator import db


def render_digest(conn, top_n: int = 20) -> str:
    scores = sorted(db.list_scores(conn), key=lambda s: s.final_score, reverse=True)[:top_n]
    lines = [f"# arXiv Digest -- {date.today().isoformat()}", ""]
    for score in scores:
        paper = db.get_paper(conn, score.arxiv_id)
        summary = db.get_summary(conn, score.arxiv_id)
        lines.append(f"## [{paper.title}]({paper.url})")
        lines.append(f"**Score:** {score.final_score:.3f}  |  **arXiv:** {paper.arxiv_id}")
        lines.append("")
        lines.append(summary.text if summary else paper.abstract)
        lines.append("")
        lines.append(f"**Why this matches:** {score.explanation}")
        lines.append("")
        lines.append(f"`arxiv-curator feedback {paper.arxiv_id} --rating up`")
        lines.append("")
    return "\n".join(lines)


def write_digest(conn, out_dir: Path, top_n: int = 20) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    content = render_digest(conn, top_n)
    dated_path = out_dir / f"{date.today().isoformat()}.md"
    dated_path.write_text(content)
    (out_dir / "latest.md").write_text(content)
    return dated_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_digest.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/arxiv_curator/digest.py tests/test_digest.py
git commit -m "Add markdown digest rendering"
```

---

### Task 11: eval.py — ranking-quality metrics and evaluation

**Files:**
- Create: `src/arxiv_curator/eval.py`
- Test: `tests/test_eval.py`

**Interfaces:**
- Consumes: `compute_centroids`, `score_paper` (Task 6); `Feedback` (Task 1); `embed_texts` (Task 5); `load_interest_profile`, `profile_to_text` (Task 3); `db.list_papers`, `db.list_feedback` (Task 2).
- Produces: `precision_at_k(ranked_ids, relevant_ids, k) -> float`, `ndcg_at_k(ranked_ids, relevant_ids, k) -> float`, `mrr(ranked_ids, relevant_ids) -> float`, `evaluate(feedback_items, vectors_by_id, interest_vector, rng_seed=42) -> dict`, `run_eval(conn, interests_path, client) -> dict`.

- [ ] **Step 1: Write the failing tests**

`tests/test_eval.py`:
```python
import numpy as np
import pytest

from arxiv_curator import db, eval as eval_module
from arxiv_curator.models import Feedback, Paper
from arxiv_curator.eval import precision_at_k, ndcg_at_k, mrr, evaluate


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
    result = eval_module.run_eval(conn, interests_path, client=None)
    assert result["status"] == "ok"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_eval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'arxiv_curator.eval'`

- [ ] **Step 3: Write implementation**

`src/arxiv_curator/eval.py`:
```python
import random

import numpy as np

from arxiv_curator import db, rank
from arxiv_curator.interests import load_interest_profile, profile_to_text
from arxiv_curator.llm.embeddings import embed_texts

MIN_FEEDBACK_FOR_EVAL = 5


def precision_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    top_k = ranked_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for aid in top_k if aid in relevant_ids)
    return hits / len(top_k)


def _dcg(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    return sum(
        1.0 / np.log2(i + 2) for i, aid in enumerate(ranked_ids[:k]) if aid in relevant_ids
    )


def ndcg_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    actual = _dcg(ranked_ids, relevant_ids, k)
    ideal_order = [aid for aid in ranked_ids if aid in relevant_ids] + \
        [aid for aid in ranked_ids if aid not in relevant_ids]
    ideal = _dcg(ideal_order, relevant_ids, k)
    return actual / ideal if ideal > 0 else 0.0


def mrr(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    for i, aid in enumerate(ranked_ids):
        if aid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate(
    feedback_items: list,
    vectors_by_id: dict[str, np.ndarray],
    interest_vector: np.ndarray,
    rng_seed: int = 42,
) -> dict:
    rated = [f for f in feedback_items if f.rating in ("up", "down")]
    if len(rated) < MIN_FEEDBACK_FOR_EVAL:
        return {"status": "insufficient_data", "rated_count": len(rated)}

    candidate_ids = list(vectors_by_id.keys())
    rng = random.Random(rng_seed)

    metrics = {
        "feedback_adjusted": {"precision_at_5": [], "precision_at_10": [], "ndcg_at_10": [], "mrr": []},
        "similarity_only_baseline": {"precision_at_5": [], "ndcg_at_10": [], "mrr": []},
        "random_baseline": {"precision_at_5": [], "ndcg_at_10": [], "mrr": []},
    }

    n_evaluated = 0
    for held_out in rated:
        if held_out.rating != "up":
            continue
        relevant = {held_out.arxiv_id}
        other_feedback = [f for f in rated if f.arxiv_id != held_out.arxiv_id]
        mean_liked, mean_disliked = rank.compute_centroids(other_feedback, vectors_by_id)

        adjusted_scored, similarity_scored = [], []
        for aid in candidate_ids:
            vec = vectors_by_id[aid]
            similarity, _, final = rank.score_paper(vec, interest_vector, mean_liked, mean_disliked)
            adjusted_scored.append((aid, final))
            similarity_scored.append((aid, similarity))

        adjusted_ranked = [aid for aid, _ in sorted(adjusted_scored, key=lambda x: x[1], reverse=True)]
        similarity_ranked = [aid for aid, _ in sorted(similarity_scored, key=lambda x: x[1], reverse=True)]
        random_ranked = candidate_ids[:]
        rng.shuffle(random_ranked)

        metrics["feedback_adjusted"]["precision_at_5"].append(precision_at_k(adjusted_ranked, relevant, 5))
        metrics["feedback_adjusted"]["precision_at_10"].append(precision_at_k(adjusted_ranked, relevant, 10))
        metrics["feedback_adjusted"]["ndcg_at_10"].append(ndcg_at_k(adjusted_ranked, relevant, 10))
        metrics["feedback_adjusted"]["mrr"].append(mrr(adjusted_ranked, relevant))

        metrics["similarity_only_baseline"]["precision_at_5"].append(precision_at_k(similarity_ranked, relevant, 5))
        metrics["similarity_only_baseline"]["ndcg_at_10"].append(ndcg_at_k(similarity_ranked, relevant, 10))
        metrics["similarity_only_baseline"]["mrr"].append(mrr(similarity_ranked, relevant))

        metrics["random_baseline"]["precision_at_5"].append(precision_at_k(random_ranked, relevant, 5))
        metrics["random_baseline"]["ndcg_at_10"].append(ndcg_at_k(random_ranked, relevant, 10))
        metrics["random_baseline"]["mrr"].append(mrr(random_ranked, relevant))
        n_evaluated += 1

    return {
        "status": "ok",
        "n_evaluated": n_evaluated,
        "feedback_adjusted": {k: _avg(v) for k, v in metrics["feedback_adjusted"].items()},
        "similarity_only_baseline": {k: _avg(v) for k, v in metrics["similarity_only_baseline"].items()},
        "random_baseline": {k: _avg(v) for k, v in metrics["random_baseline"].items()},
    }


def run_eval(conn, interests_path, client) -> dict:
    profile = load_interest_profile(interests_path)
    interest_vector = embed_texts([profile_to_text(profile)], client)[0]
    papers = db.list_papers(conn)
    vectors = embed_texts([p.abstract for p in papers], client) if papers else np.empty((0, 0))
    vectors_by_id = {p.arxiv_id: vec for p, vec in zip(papers, vectors)}
    feedback_items = db.list_feedback(conn)
    return evaluate(feedback_items, vectors_by_id, interest_vector)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_eval.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add src/arxiv_curator/eval.py tests/test_eval.py
git commit -m "Add leave-one-out ranking-quality eval with baselines"
```

---

### Task 12: cli.py — Typer app wiring everything together

**Files:**
- Create: `src/arxiv_curator/cli.py`
- Create: `interests.yaml` (starter content)
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: every module from Tasks 1-11.
- Produces: Typer `app` with commands `fetch`, `summarize`, `rank`, `show`, `feedback`, `digest`, `eval`, `run`.

- [ ] **Step 1: Write the failing tests**

These test the commands that don't require Gemini (`feedback`, `show`, `digest`), using `CliRunner` against a temp db. `fetch`/`summarize`/`rank`/`eval`/`run` need real network/API access and are covered by manual verification in Task 13, not automated tests.

`tests/test_cli.py`:
```python
from typer.testing import CliRunner

from arxiv_curator import cli, db
from arxiv_curator.models import Paper

runner = CliRunner()


def seed_db(db_path):
    conn = db.get_connection(db_path)
    db.init_db(conn)
    db.insert_paper(conn, Paper(
        arxiv_id="2601.00001", title="A Great Paper", authors="Ada Author",
        abstract="An abstract about transformers.", categories="cs.AI",
        published="2026-01-01T00:00:00Z", url="https://arxiv.org/abs/2601.00001",
    ))
    conn.close()


def test_feedback_command_records_rating(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    seed_db(db_path)
    monkeypatch.setattr(cli, "DB_PATH", db_path)

    result = runner.invoke(cli.app, ["feedback", "2601.00001", "--rating", "up"])
    assert result.exit_code == 0

    conn = db.get_connection(db_path)
    items = db.list_feedback(conn)
    assert items[0].rating == "up"


def test_feedback_command_rejects_invalid_rating(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    seed_db(db_path)
    monkeypatch.setattr(cli, "DB_PATH", db_path)

    result = runner.invoke(cli.app, ["feedback", "2601.00001", "--rating", "sideways"])
    assert result.exit_code != 0


def test_show_command_prints_title_and_abstract(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    seed_db(db_path)
    monkeypatch.setattr(cli, "DB_PATH", db_path)

    result = runner.invoke(cli.app, ["show", "2601.00001"])
    assert result.exit_code == 0
    assert "A Great Paper" in result.output


def test_digest_command_writes_file(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    seed_db(db_path)
    monkeypatch.setattr(cli, "DB_PATH", db_path)
    digests_dir = tmp_path / "digests"
    monkeypatch.setattr(cli, "DIGESTS_DIR", digests_dir)

    conn = db.get_connection(db_path)
    from arxiv_curator.models import Score
    db.upsert_score(conn, Score(
        arxiv_id="2601.00001", similarity=0.9, feedback_adjustment=0.0,
        final_score=0.9, explanation="Matches.", created_at="t",
    ))
    conn.close()

    result = runner.invoke(cli.app, ["digest"])
    assert result.exit_code == 0
    assert (digests_dir / "latest.md").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'arxiv_curator.cli'`

- [ ] **Step 3: Write implementation**

`src/arxiv_curator/cli.py`:
```python
from datetime import datetime, timezone
from pathlib import Path

import typer
from dotenv import load_dotenv

from arxiv_curator import db
from arxiv_curator import digest as digest_module
from arxiv_curator import eval as eval_module
from arxiv_curator import feedback as feedback_module
from arxiv_curator import fetch as fetch_module
from arxiv_curator import rank as rank_module
from arxiv_curator.llm import factory
from arxiv_curator.llm.gemini_provider import GeminiProvider
from arxiv_curator.models import Summary

load_dotenv()

app = typer.Typer()

DB_PATH = Path("data/arxiv_curator.db")
INTERESTS_PATH = Path("interests.yaml")
DIGESTS_DIR = Path("digests")
DEFAULT_CATEGORIES = "cs.AI,cs.LG,cs.CL,stat.ML"


def get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = db.get_connection(DB_PATH)
    db.init_db(conn)
    return conn


@app.command()
def fetch(
    categories: str = typer.Option(DEFAULT_CATEGORIES),
    max_results: int = typer.Option(100),
):
    conn = get_conn()
    count = fetch_module.fetch_and_store(conn, categories.split(","), max_results)
    typer.echo(f"Fetched {count} new papers.")


@app.command()
def summarize(limit: int = typer.Option(50)):
    conn = get_conn()
    client = factory.get_client()
    provider = GeminiProvider(client)
    for paper in db.papers_missing_summary(conn)[:limit]:
        text = provider.summarize(paper)
        db.insert_summary(conn, Summary(
            arxiv_id=paper.arxiv_id, text=text,
            created_at=datetime.now(timezone.utc).isoformat(),
        ))
        typer.echo(f"Summarized {paper.arxiv_id}")


@app.command()
def rank(top: int = typer.Option(20)):
    conn = get_conn()
    client = factory.get_client()
    provider = GeminiProvider(client)
    scores = rank_module.rank_papers(conn, INTERESTS_PATH, provider, client)
    scores.sort(key=lambda s: s.final_score, reverse=True)
    for s in scores[:top]:
        typer.echo(f"{s.final_score:.3f}  {s.arxiv_id}")


@app.command()
def show(arxiv_id: str):
    conn = get_conn()
    paper = db.get_paper(conn, arxiv_id)
    if paper is None:
        typer.echo(f"No such paper: {arxiv_id}")
        raise typer.Exit(code=1)
    summary = db.get_summary(conn, arxiv_id)
    score = db.get_score(conn, arxiv_id)
    typer.echo(f"# {paper.title}\n{paper.url}\n")
    typer.echo(summary.text if summary else paper.abstract)
    if score is not None:
        typer.echo(f"\nScore: {score.final_score:.3f}")
        typer.echo(f"Why this matches: {score.explanation}")


@app.command()
def feedback(
    arxiv_id: str,
    rating: str = typer.Option(None),
    note: str = typer.Option(None),
    pages_read: int = typer.Option(None),
    total_pages: int = typer.Option(None),
):
    conn = get_conn()
    try:
        feedback_module.record_feedback(
            conn, arxiv_id, rating=rating, note=note,
            pages_read=pages_read, total_pages=total_pages,
        )
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1)
    typer.echo(f"Recorded feedback for {arxiv_id}")


@app.command()
def digest(top: int = typer.Option(20)):
    conn = get_conn()
    path = digest_module.write_digest(conn, DIGESTS_DIR, top)
    typer.echo(f"Wrote {path}")


@app.command(name="eval")
def eval_cmd():
    conn = get_conn()
    client = factory.get_client()
    result = eval_module.run_eval(conn, INTERESTS_PATH, client)
    typer.echo(result)


@app.command()
def run():
    conn = get_conn()
    client = factory.get_client()
    provider = GeminiProvider(client)
    fetch_module.fetch_and_store(conn, DEFAULT_CATEGORIES.split(","), 100)
    for paper in db.papers_missing_summary(conn):
        text = provider.summarize(paper)
        db.insert_summary(conn, Summary(
            arxiv_id=paper.arxiv_id, text=text,
            created_at=datetime.now(timezone.utc).isoformat(),
        ))
    rank_module.rank_papers(conn, INTERESTS_PATH, provider, client)
    path = digest_module.write_digest(conn, DIGESTS_DIR, 20)
    typer.echo(f"Wrote {path}")


if __name__ == "__main__":
    app()
```

`interests.yaml` (starter content, meant to be edited):
```yaml
summary: >
  I'm interested in practical, applied AI systems: retrieval-augmented
  generation, agent architectures, evaluation methodology for LLMs, and
  recommender/ranking systems. I prefer papers with real experiments over
  pure theory.
topics:
  - retrieval-augmented generation
  - agent evaluation
  - recommender systems
  - LLM evaluation
keywords:
  - RAG
  - agents
  - evaluation
  - ranking
  - embeddings
liked_examples:
  - "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cli.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/arxiv_curator/cli.py interests.yaml tests/test_cli.py
git commit -m "Wire up Typer CLI and add starter interests.yaml"
```

---

### Task 13: README, CI workflow, and scheduled digest workflow

**Files:**
- Create: `README.md`
- Create: `digests/.gitkeep`
- Create: `.github/workflows/ci.yml`
- Create: `.github/workflows/daily-digest.yml`

**Interfaces:**
- Consumes: `arxiv-curator` console script (Task 12) as the command the workflows invoke.

- [ ] **Step 1: Write `README.md`**

```markdown
# arXiv Curator

Fetches newest arXiv papers, summarizes them, ranks them against your
interests, learns from your feedback (including how much you actually
read), explains why each paper matched, and evaluates ranking quality.

## Setup

    python -m venv .venv
    source .venv/bin/activate
    pip install -e ".[dev]"
    cp .env.example .env   # add your GEMINI_API_KEY

Edit `interests.yaml` to describe what you actually care about.

## Usage

    arxiv-curator fetch --categories cs.AI,cs.LG,cs.CL,stat.ML
    arxiv-curator summarize
    arxiv-curator rank
    arxiv-curator show <arxiv_id>
    arxiv-curator feedback <arxiv_id> --rating up
    arxiv-curator feedback <arxiv_id> --pages-read 5 --total-pages 12
    arxiv-curator digest
    arxiv-curator eval
    arxiv-curator run   # fetch + summarize + rank + digest in one shot

## How ranking works

Papers are embedded (Gemini embeddings) and scored by cosine similarity
to your `interests.yaml` profile. As you rate papers up/down (optionally
with how much you actually read), a Rocchio-style centroid of liked and
disliked papers nudges future scores toward what you've responded to.
Each ranked paper gets a grounded "why this matches" explanation, built
from real signals (overlapping keywords, closest liked paper) rather
than free-floating LLM justification.

## Viewing digests

`digests/latest.md` always has the most recent run; `digests/YYYY-MM-DD.md`
keeps history. Both render natively when browsing this repo on GitHub.

## Evals

`arxiv-curator eval` runs a leave-one-out evaluation against your stored
feedback (Precision@5/@10, NDCG@10, MRR), compared against a
similarity-only baseline and a random baseline. Needs at least 5 rated
papers to produce real numbers.

## Tests

    pytest
```

- [ ] **Step 2: Create `digests/.gitkeep`**

Empty file so the directory is tracked before any digest exists.

- [ ] **Step 3: Write `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev]"
      - run: pytest
```

- [ ] **Step 4: Write `.github/workflows/daily-digest.yml`**

```yaml
name: Daily Digest

on:
  schedule:
    - cron: "0 13 * * *"
  workflow_dispatch:

jobs:
  digest:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Restore database cache
        uses: actions/cache@v4
        with:
          path: data/arxiv_curator.db
          key: arxiv-curator-db-${{ github.run_id }}
          restore-keys: |
            arxiv-curator-db-
      - run: pip install -e .
      - run: arxiv-curator run
        env:
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
      - uses: stefanzweifel/git-auto-commit-action@v5
        with:
          commit_message: "Daily arXiv digest"
          file_pattern: "digests/*.md"
```

- [ ] **Step 5: Commit**

```bash
git add README.md digests/.gitkeep .github/workflows/ci.yml .github/workflows/daily-digest.yml
git commit -m "Add README and CI/scheduled-digest workflows"
```

---

### Task 14: End-to-end manual verification

No new files. This exercises the pieces automated tests can't cover (real arXiv network call, real Gemini API).

- [ ] **Step 1: Run the full local test suite**

Run: `pytest -v`
Expected: all tests from Tasks 1-13 pass (roughly 45+ tests total).

- [ ] **Step 2: Fetch real papers**

Run: `arxiv-curator fetch --categories cs.AI --max-results 10`
Expected: `Fetched <=10 new papers.` and `data/arxiv_curator.db` now exists.

- [ ] **Step 3: Add your `GEMINI_API_KEY` to `.env` and summarize**

Run: `arxiv-curator summarize --limit 10`
Expected: one `Summarized <arxiv_id>` line per fetched paper, no errors.

- [ ] **Step 4: Edit `interests.yaml` to reflect your real interests, then rank**

Run: `arxiv-curator rank`
Expected: a list of `<score>  <arxiv_id>` lines, sorted descending by score.

- [ ] **Step 5: Inspect one paper's explanation**

Run: `arxiv-curator show <an arxiv_id from step 4>`
Expected: title, summary, score, and a "Why this matches" line that references real keywords/topics from your `interests.yaml`, not generic filler.

- [ ] **Step 6: Record feedback and re-rank**

Run: `arxiv-curator feedback <arxiv_id> --rating up --pages-read 8 --total-pages 12` then `arxiv-curator rank`
Expected: no errors; over time (after 5+ rated papers) scores should visibly shift toward papers similar to what you rated up.

- [ ] **Step 7: Generate a digest and confirm it renders on GitHub**

Run: `arxiv-curator digest`
Expected: `digests/YYYY-MM-DD.md` and `digests/latest.md` written. Push to a GitHub repo and open the file in the browser at `github.com/<you>/<repo>/blob/main/digests/latest.md` — confirm it renders as formatted markdown, not raw text.

- [ ] **Step 8: Run eval with insufficient data, then with enough**

Run: `arxiv-curator eval`
Expected: with fewer than 5 rated papers, `{'status': 'insufficient_data', ...}`. After rating 5+ papers, expect `{'status': 'ok', 'feedback_adjusted': {...}, 'similarity_only_baseline': {...}, 'random_baseline': {...}}`.

- [ ] **Step 9: Confirm CI is green**

Push the branch, open a PR (or push to main), and confirm the `CI` GitHub Action passes without any secrets configured.

- [ ] **Step 10: Manually trigger the scheduled workflow once**

In the GitHub Actions tab, run `Daily Digest` via `workflow_dispatch` (after adding `GEMINI_API_KEY` as a repo secret). Expected: a new commit appears with an updated `digests/latest.md`, and re-running it again dedups correctly against the cached db instead of re-adding the same papers.
