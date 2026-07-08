import json
import sqlite3
from datetime import datetime, timezone
from typing import Optional

import numpy as np

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

CREATE TABLE IF NOT EXISTS embeddings (
    arxiv_id TEXT PRIMARY KEY REFERENCES papers(arxiv_id),
    vector TEXT NOT NULL,
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


def upsert_embedding(conn, arxiv_id: str, vector) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO embeddings (arxiv_id, vector, created_at) VALUES (?, ?, ?)",
        (arxiv_id, json.dumps([float(x) for x in vector]), datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


def get_embeddings(conn, arxiv_ids: list[str]) -> dict:
    if not arxiv_ids:
        return {}
    placeholders = ",".join("?" for _ in arxiv_ids)
    rows = conn.execute(
        f"SELECT arxiv_id, vector FROM embeddings WHERE arxiv_id IN ({placeholders})",
        arxiv_ids,
    ).fetchall()
    return {row["arxiv_id"]: np.array(json.loads(row["vector"])) for row in rows}
