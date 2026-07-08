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
