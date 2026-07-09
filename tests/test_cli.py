from pathlib import Path

from typer.testing import CliRunner

from arxiv_curator import cli, db
from arxiv_curator.models import Paper

runner = CliRunner()


def test_resolve_data_dir_uses_env_var(monkeypatch):
    monkeypatch.setenv("ARXIV_CURATOR_DATA_DIR", "/tmp/custom-arxiv-data")
    assert cli._resolve_data_dir() == Path("/tmp/custom-arxiv-data")


def test_resolve_data_dir_defaults_to_home_dir(monkeypatch):
    monkeypatch.delenv("ARXIV_CURATOR_DATA_DIR", raising=False)
    assert cli._resolve_data_dir() == Path("~/arxiv-curator-data").expanduser()


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


def test_show_command_omits_explanation_when_not_yet_generated(tmp_path, monkeypatch):
    from arxiv_curator.models import Score

    db_path = tmp_path / "test.db"
    seed_db(db_path)
    monkeypatch.setattr(cli, "DB_PATH", db_path)

    conn = db.get_connection(db_path)
    db.upsert_score(conn, Score(
        arxiv_id="2601.00001", similarity=0.5, feedback_adjustment=0.0,
        final_score=0.5, explanation="", created_at="t",
    ))
    conn.close()

    result = runner.invoke(cli.app, ["show", "2601.00001"])
    assert result.exit_code == 0
    assert "Score: 0.500" in result.output
    assert "Why this matches" not in result.output


def test_show_command_prints_explanation_when_present(tmp_path, monkeypatch):
    from arxiv_curator.models import Score

    db_path = tmp_path / "test.db"
    seed_db(db_path)
    monkeypatch.setattr(cli, "DB_PATH", db_path)

    conn = db.get_connection(db_path)
    db.upsert_score(conn, Score(
        arxiv_id="2601.00001", similarity=0.5, feedback_adjustment=0.0,
        final_score=0.5, explanation="Matches your interests.", created_at="t",
    ))
    conn.close()

    result = runner.invoke(cli.app, ["show", "2601.00001"])
    assert result.exit_code == 0
    assert "Why this matches: Matches your interests." in result.output


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


def test_add_command_inserts_new_paper(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    conn = db.get_connection(db_path)
    db.init_db(conn)
    conn.close()
    monkeypatch.setattr(cli, "DB_PATH", db_path)

    fake_paper = Paper(
        arxiv_id="2606.06036v1", title="Memory is Reconstructed, Not Retrieved",
        authors="Shuo Ji", abstract="An abstract about agent memory.",
        categories="cs.AI", published="2026-06-04T00:00:00Z",
        url="https://arxiv.org/abs/2606.06036v1",
    )
    monkeypatch.setattr(cli.fetch_module, "fetch_paper_by_id", lambda arxiv_id: fake_paper)

    result = runner.invoke(cli.app, ["add", "2606.06036"])
    assert result.exit_code == 0
    assert "Memory is Reconstructed" in result.output

    conn = db.get_connection(db_path)
    assert db.get_paper(conn, "2606.06036v1") is not None


def test_add_command_reports_when_already_present(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    seed_db(db_path)
    monkeypatch.setattr(cli, "DB_PATH", db_path)
    monkeypatch.setattr(
        cli.fetch_module, "fetch_paper_by_id",
        lambda arxiv_id: Paper(
            arxiv_id="2601.00001", title="A Great Paper", authors="Ada Author",
            abstract="An abstract about transformers.", categories="cs.AI",
            published="2026-01-01T00:00:00Z", url="https://arxiv.org/abs/2601.00001",
        ),
    )

    result = runner.invoke(cli.app, ["add", "2601.00001"])
    assert result.exit_code == 0
    assert "Already in your database" in result.output


def test_add_command_reports_not_found(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    conn = db.get_connection(db_path)
    db.init_db(conn)
    conn.close()
    monkeypatch.setattr(cli, "DB_PATH", db_path)
    monkeypatch.setattr(cli.fetch_module, "fetch_paper_by_id", lambda arxiv_id: None)

    result = runner.invoke(cli.app, ["add", "9999.99999"])
    assert result.exit_code == 1
    assert "No such arXiv paper" in result.output


def test_add_command_inserts_paper_with_manual_source(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    conn = db.get_connection(db_path)
    db.init_db(conn)
    conn.close()
    monkeypatch.setattr(cli, "DB_PATH", db_path)

    fake_paper = Paper(
        arxiv_id="2606.06036v1", title="Memory is Reconstructed, Not Retrieved",
        authors="Shuo Ji", abstract="An abstract about agent memory.",
        categories="cs.AI", published="2026-06-04T00:00:00Z",
        url="https://arxiv.org/abs/2606.06036v1",
    )
    monkeypatch.setattr(cli.fetch_module, "fetch_paper_by_id", lambda arxiv_id: fake_paper)

    result = runner.invoke(cli.app, ["add", "2606.06036"])
    assert result.exit_code == 0

    conn = db.get_connection(db_path)
    row = conn.execute(
        "SELECT source FROM papers WHERE arxiv_id = ?", ("2606.06036v1",)
    ).fetchone()
    assert row["source"] == "manual"


def test_summarize_command_fails_cleanly_without_api_key(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    seed_db(db_path)
    monkeypatch.setattr(cli, "DB_PATH", db_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    result = runner.invoke(cli.app, ["summarize"])

    assert result.exit_code == 1
    assert "GEMINI_API_KEY" in result.output
    assert not isinstance(result.exception, RuntimeError)


def test_run_command_only_summarizes_papers_that_make_the_digest(tmp_path, monkeypatch):
    from arxiv_curator.models import Score

    db_path = tmp_path / "test.db"
    conn = db.get_connection(db_path)
    db.init_db(conn)
    for arxiv_id, title in [("2601.00001", "Top Paper"), ("2601.00002", "Bottom Paper")]:
        db.insert_paper(conn, Paper(
            arxiv_id=arxiv_id, title=title, authors="A", abstract="B",
            categories="cs.AI", published="2026-01-01T00:00:00Z",
            url=f"https://arxiv.org/abs/{arxiv_id}",
        ))
    conn.close()

    monkeypatch.setattr(cli, "DB_PATH", db_path)
    monkeypatch.setattr(cli, "DIGESTS_DIR", tmp_path / "digests")
    monkeypatch.setattr(cli, "DEFAULT_DIGEST_TOP_N", 1)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(cli.factory, "get_client", lambda: "fake-client")
    monkeypatch.setattr(cli.fetch_module, "fetch_and_store", lambda *a, **k: 0)

    def fake_rank_papers(conn, interests_path, client):
        db.upsert_score(conn, Score(
            arxiv_id="2601.00001", similarity=0.9, feedback_adjustment=0.0,
            final_score=0.9, explanation="Top match.", created_at="t",
        ))
        db.upsert_score(conn, Score(
            arxiv_id="2601.00002", similarity=0.1, feedback_adjustment=0.0,
            final_score=0.1, explanation="Weak match.", created_at="t",
        ))
        return []

    monkeypatch.setattr(cli.rank_module, "rank_papers", fake_rank_papers)
    monkeypatch.setattr(cli.rank_module, "explain_papers", lambda *a, **k: None)

    summarized_ids = []

    class FakeProvider:
        def __init__(self, client):
            pass

        def summarize(self, paper):
            summarized_ids.append(paper.arxiv_id)
            return f"Summary of {paper.arxiv_id}"

    monkeypatch.setattr(cli, "GeminiProvider", FakeProvider)

    result = runner.invoke(cli.app, ["run"])
    assert result.exit_code == 0
    assert summarized_ids == ["2601.00001"]

    conn = db.get_connection(db_path)
    assert db.get_summary(conn, "2601.00001") is not None
    assert db.get_summary(conn, "2601.00002") is None


def test_sync_command_reports_result(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "DATA_DIR", tmp_path)
    monkeypatch.setattr(cli.sync_module, "sync", lambda data_dir: "up-to-date")

    result = runner.invoke(cli.app, ["sync"])
    assert result.exit_code == 0
    assert "up-to-date" in result.output


def test_sync_command_fails_cleanly_on_sync_error(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "DATA_DIR", tmp_path)

    def raise_sync_error(data_dir):
        raise cli.sync_module.SyncError("push rejected -- re-run sync")

    monkeypatch.setattr(cli.sync_module, "sync", raise_sync_error)

    result = runner.invoke(cli.app, ["sync"])
    assert result.exit_code == 1
    assert "push rejected" in result.output


def test_agent_pick_command_writes_digest_with_picks(tmp_path, monkeypatch):
    from arxiv_curator.models import AgentPickDecision

    db_path = tmp_path / "test.db"
    seed_db(db_path)
    monkeypatch.setattr(cli, "DB_PATH", db_path)
    digests_dir = tmp_path / "digests"
    monkeypatch.setattr(cli, "DIGESTS_DIR", digests_dir)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(cli.factory, "get_client", lambda: "fake-client")
    monkeypatch.setattr(
        cli.agent_pick_module, "run_agent_pick",
        lambda conn, client: [
            AgentPickDecision(arxiv_id="2601.00001", status="picked", reasoning="great fit", decided_at="t"),
        ],
    )

    result = runner.invoke(cli.app, ["agent-pick"])
    assert result.exit_code == 0
    assert (digests_dir / "agent-pick-latest.md").exists()
    assert "great fit" in (digests_dir / "agent-pick-latest.md").read_text()


def test_agent_pick_command_reports_when_nothing_clears_bar(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    seed_db(db_path)
    monkeypatch.setattr(cli, "DB_PATH", db_path)
    digests_dir = tmp_path / "digests"
    monkeypatch.setattr(cli, "DIGESTS_DIR", digests_dir)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(cli.factory, "get_client", lambda: "fake-client")
    monkeypatch.setattr(cli.agent_pick_module, "run_agent_pick", lambda conn, client: [])

    result = runner.invoke(cli.app, ["agent-pick"])
    assert result.exit_code == 0
    assert "nothing cleared the bar" in result.output


def test_agent_pick_command_fails_cleanly_without_api_key(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    seed_db(db_path)
    monkeypatch.setattr(cli, "DB_PATH", db_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    result = runner.invoke(cli.app, ["agent-pick"])
    assert result.exit_code == 1
    assert "GEMINI_API_KEY" in result.output
