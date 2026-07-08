from datetime import datetime, timedelta, timezone
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
DEFAULT_DIGEST_WINDOW_DAYS = 2


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
def digest(top: int = typer.Option(20), since_days: int = typer.Option(DEFAULT_DIGEST_WINDOW_DAYS)):
    conn = get_conn()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=since_days)).isoformat()
    path = digest_module.write_digest(conn, DIGESTS_DIR, top, since=cutoff)
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
    cutoff = (datetime.now(timezone.utc) - timedelta(days=DEFAULT_DIGEST_WINDOW_DAYS)).isoformat()
    path = digest_module.write_digest(conn, DIGESTS_DIR, 20, since=cutoff)
    typer.echo(f"Wrote {path}")


if __name__ == "__main__":
    app()
