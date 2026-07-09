import numpy as np
from datetime import date

from arxiv_curator import agent_pick, db, rank
from arxiv_curator.models import AgentPickDecision, Feedback, Paper, Summary


def make_paper(arxiv_id, abstract):
    return Paper(
        arxiv_id=arxiv_id, title=f"Title {arxiv_id}", authors="Author",
        abstract=abstract, categories="cs.AI",
        published="2026-01-01T00:00:00Z", url=f"https://arxiv.org/abs/{arxiv_id}",
    )


def make_conn():
    conn = db.get_connection(":memory:")
    db.init_db(conn)
    return conn


def test_build_shortlist_ranks_new_papers_by_criteria_similarity(monkeypatch):
    conn = make_conn()
    db.insert_paper(conn, make_paper("relevant1", "This paper is about agent tool use."))
    db.insert_paper(conn, make_paper("irrelevant1", "This paper is about gardening."))

    def fake_embed_texts(texts, client):
        return np.array([[1.0, 0.0] if "agent" in t.lower() else [0.0, 1.0] for t in texts])

    monkeypatch.setattr(agent_pick, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(rank, "embed_texts", fake_embed_texts)

    shortlist, held_reasoning = agent_pick.build_shortlist(
        conn, client=None, criteria_text="agent infrastructure", limit=10,
    )
    assert [p.arxiv_id for p in shortlist] == ["relevant1", "irrelevant1"]
    assert held_reasoning == {}


def test_build_shortlist_respects_limit(monkeypatch):
    conn = make_conn()
    for i in range(3):
        db.insert_paper(conn, make_paper(f"p{i}", f"paper {i} about agents"))

    def fake_embed_texts(texts, client):
        return np.array([[1.0, 0.0] for _ in texts])

    monkeypatch.setattr(agent_pick, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(rank, "embed_texts", fake_embed_texts)
    shortlist, _ = agent_pick.build_shortlist(conn, client=None, criteria_text="agents", limit=2)
    assert len(shortlist) == 2


def test_build_shortlist_always_includes_held_papers(monkeypatch):
    conn = make_conn()
    db.insert_paper(conn, make_paper("held1", "unrelated abstract about gardening"))
    db.upsert_agent_pick_decision(conn, AgentPickDecision(
        arxiv_id="held1", status="held", reasoning="worth a second look", decided_at="t",
    ))

    def fake_embed_texts(texts, client):
        return np.array([[0.0, 1.0] for _ in texts])

    monkeypatch.setattr(agent_pick, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(rank, "embed_texts", fake_embed_texts)
    shortlist, held_reasoning = agent_pick.build_shortlist(
        conn, client=None, criteria_text="agents", limit=10,
    )
    assert [p.arxiv_id for p in shortlist] == ["held1"]
    assert held_reasoning["held1"] == "worth a second look"


def test_build_shortlist_excludes_rejected_papers(monkeypatch):
    conn = make_conn()
    db.insert_paper(conn, make_paper("rejected1", "already rejected paper about agents"))
    db.upsert_agent_pick_decision(conn, AgentPickDecision(
        arxiv_id="rejected1", status="rejected", reasoning="not interesting", decided_at="t",
    ))

    def fake_embed_texts(texts, client):
        return np.array([[1.0, 0.0] for _ in texts])

    monkeypatch.setattr(agent_pick, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(rank, "embed_texts", fake_embed_texts)
    shortlist, _ = agent_pick.build_shortlist(conn, client=None, criteria_text="agents", limit=10)
    assert shortlist == []


def test_get_paper_detail_returns_full_fields():
    conn = make_conn()
    db.insert_paper(conn, make_paper("p1", "An abstract about agents."))
    db.insert_summary(conn, Summary(arxiv_id="p1", text="A short summary.", created_at="t"))

    detail = agent_pick.get_paper_detail(conn, "p1")
    assert detail["title"] == "Title p1"
    assert detail["summary"] == "A short summary."


def test_get_paper_detail_summary_is_none_when_missing():
    conn = make_conn()
    db.insert_paper(conn, make_paper("p1", "An abstract."))
    detail = agent_pick.get_paper_detail(conn, "p1")
    assert detail["summary"] is None


def test_get_paper_detail_returns_error_for_unknown_paper():
    conn = make_conn()
    detail = agent_pick.get_paper_detail(conn, "unknown1")
    assert "error" in detail


def test_get_feedback_history_returns_top_k_most_similar_rated_papers():
    conn = make_conn()
    for arxiv_id in ["candidate1", "close1", "close2", "far1"]:
        db.insert_paper(conn, make_paper(arxiv_id, "abstract"))
    db.upsert_embedding(conn, "candidate1", [1.0, 0.0])
    db.upsert_embedding(conn, "close1", [0.9, 0.1])
    db.upsert_embedding(conn, "close2", [0.8, 0.2])
    db.upsert_embedding(conn, "far1", [0.0, 1.0])
    db.insert_feedback(conn, Feedback(arxiv_id="close1", created_at="t", rating="up", note="loved it"))
    db.insert_feedback(conn, Feedback(arxiv_id="close2", created_at="t", rating="down"))
    db.insert_feedback(conn, Feedback(arxiv_id="far1", created_at="t", rating="up"))

    result = agent_pick.get_feedback_history(conn, client=None, arxiv_id="candidate1", top_k=2)
    ids = [entry["arxiv_id"] for entry in result["similar_rated_papers"]]
    assert ids == ["close1", "close2"]
    assert result["similar_rated_papers"][0]["note"] == "loved it"


def test_get_feedback_history_excludes_unrated_feedback():
    conn = make_conn()
    db.insert_paper(conn, make_paper("candidate1", "abstract"))
    db.insert_paper(conn, make_paper("note_only1", "abstract"))
    db.upsert_embedding(conn, "candidate1", [1.0, 0.0])
    db.upsert_embedding(conn, "note_only1", [1.0, 0.0])
    db.insert_feedback(conn, Feedback(arxiv_id="note_only1", created_at="t", note="read half of it"))

    result = agent_pick.get_feedback_history(conn, client=None, arxiv_id="candidate1")
    assert result["similar_rated_papers"] == []


def test_get_feedback_history_returns_empty_when_candidate_has_no_embedding():
    conn = make_conn()
    db.insert_paper(conn, make_paper("candidate1", "abstract"))
    result = agent_pick.get_feedback_history(conn, client=None, arxiv_id="candidate1")
    assert result["similar_rated_papers"] == []


def test_validate_decisions_accepts_full_valid_coverage():
    decisions = [
        {"arxiv_id": "p1", "status": "picked", "reasoning": "great fit"},
        {"arxiv_id": "p2", "status": "held", "reasoning": "borderline"},
    ]
    result = agent_pick.validate_decisions({"p1", "p2"}, decisions)
    assert result == decisions


def test_validate_decisions_rejects_unknown_arxiv_id():
    import pytest
    with pytest.raises(agent_pick.InvalidFinalizePayload, match="not in the shortlist"):
        agent_pick.validate_decisions({"p1"}, [{"arxiv_id": "unknown1", "status": "picked", "reasoning": "x"}])


def test_validate_decisions_rejects_missing_coverage():
    import pytest
    with pytest.raises(agent_pick.InvalidFinalizePayload, match="missing decisions"):
        agent_pick.validate_decisions({"p1", "p2"}, [{"arxiv_id": "p1", "status": "held", "reasoning": "x"}])


def test_validate_decisions_rejects_too_many_picks():
    import pytest
    ids = {f"p{i}" for i in range(4)}
    decisions = [{"arxiv_id": f"p{i}", "status": "picked", "reasoning": "x"} for i in range(4)]
    with pytest.raises(agent_pick.InvalidFinalizePayload, match="max is 3"):
        agent_pick.validate_decisions(ids, decisions)


def test_validate_decisions_rejects_invalid_status():
    import pytest
    with pytest.raises(agent_pick.InvalidFinalizePayload, match="invalid status"):
        agent_pick.validate_decisions({"p1"}, [{"arxiv_id": "p1", "status": "maybe", "reasoning": "x"}])


def test_validate_decisions_rejects_missing_reasoning():
    import pytest
    with pytest.raises(agent_pick.InvalidFinalizePayload, match="missing reasoning"):
        agent_pick.validate_decisions({"p1"}, [{"arxiv_id": "p1", "status": "held", "reasoning": ""}])


def test_validate_decisions_rejects_duplicate_decision_for_same_paper():
    import pytest
    decisions = [
        {"arxiv_id": "p1", "status": "held", "reasoning": "x"},
        {"arxiv_id": "p1", "status": "rejected", "reasoning": "y"},
    ]
    with pytest.raises(agent_pick.InvalidFinalizePayload, match="duplicate"):
        agent_pick.validate_decisions({"p1"}, decisions)


def test_run_agent_pick_persists_validated_decisions(monkeypatch):
    conn = make_conn()
    db.insert_paper(conn, make_paper("p1", "about agent tool use"))

    def fake_embed_texts(texts, client):
        return np.array([[1.0, 0.0] for _ in texts])

    monkeypatch.setattr(agent_pick, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(rank, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(
        agent_pick, "run_tool_loop",
        lambda *a, **k: {"decisions": [{"arxiv_id": "p1", "status": "picked", "reasoning": "great fit"}]},
    )

    decisions = agent_pick.run_agent_pick(conn, client=None)
    assert len(decisions) == 1
    assert decisions[0].status == "picked"
    assert db.get_agent_pick_decision(conn, "p1").status == "picked"


def test_run_agent_pick_returns_empty_list_when_shortlist_is_empty(monkeypatch):
    conn = make_conn()

    loop_called = []
    monkeypatch.setattr(agent_pick, "run_tool_loop", lambda *a, **k: loop_called.append(1))

    decisions = agent_pick.run_agent_pick(conn, client=None)
    assert decisions == []
    assert loop_called == []


def test_run_agent_pick_raises_and_persists_nothing_on_invalid_finalize_payload(monkeypatch):
    conn = make_conn()
    db.insert_paper(conn, make_paper("p1", "about agent tool use"))

    def fake_embed_texts(texts, client):
        return np.array([[1.0, 0.0] for _ in texts])

    monkeypatch.setattr(agent_pick, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(rank, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(
        agent_pick, "run_tool_loop",
        lambda *a, **k: {"decisions": [{"arxiv_id": "unknown1", "status": "picked", "reasoning": "x"}]},
    )

    import pytest
    with pytest.raises(agent_pick.InvalidFinalizePayload):
        agent_pick.run_agent_pick(conn, client=None)
    assert db.get_agent_pick_decision(conn, "p1") is None


def test_render_agent_pick_digest_includes_title_summary_and_reasoning():
    conn = make_conn()
    db.insert_paper(conn, make_paper("p1", "An abstract."))
    db.insert_summary(conn, Summary(arxiv_id="p1", text="A short summary.", created_at="t"))

    text = agent_pick.render_agent_pick_digest(conn, [
        AgentPickDecision(arxiv_id="p1", status="picked", reasoning="Great fit for career growth.", decided_at="t"),
    ])
    assert "Title p1" in text
    assert "A short summary." in text
    assert "Great fit for career growth." in text
    assert "arxiv-curator feedback p1 --rating up" in text


def test_render_agent_pick_digest_falls_back_to_abstract_without_summary():
    conn = make_conn()
    db.insert_paper(conn, make_paper("p1", "The raw abstract text."))

    text = agent_pick.render_agent_pick_digest(conn, [
        AgentPickDecision(arxiv_id="p1", status="picked", reasoning="x", decided_at="t"),
    ])
    assert "The raw abstract text." in text


def test_render_agent_pick_digest_explains_when_nothing_picked():
    conn = make_conn()
    text = agent_pick.render_agent_pick_digest(conn, [])
    assert "Nothing cleared the bar this run." in text


def test_render_agent_pick_digest_skips_decision_with_missing_paper():
    conn = make_conn()
    db.insert_paper(conn, make_paper("p1", "An abstract."))

    text = agent_pick.render_agent_pick_digest(conn, [
        AgentPickDecision(arxiv_id="missing1", status="picked", reasoning="ghost", decided_at="t"),
        AgentPickDecision(arxiv_id="p1", status="picked", reasoning="x", decided_at="t"),
    ])
    assert "missing1" not in text
    assert "ghost" not in text
    assert "Title p1" in text


def test_render_agent_pick_digest_explains_when_all_papers_missing():
    conn = make_conn()

    text = agent_pick.render_agent_pick_digest(conn, [
        AgentPickDecision(arxiv_id="missing1", status="picked", reasoning="ghost", decided_at="t"),
        AgentPickDecision(arxiv_id="missing2", status="picked", reasoning="ghost2", decided_at="t"),
    ])
    assert "Nothing cleared the bar this run." in text
    assert "missing1" not in text
    assert "missing2" not in text


def test_write_agent_pick_digest_creates_dated_and_latest_files(tmp_path):
    conn = make_conn()
    db.insert_paper(conn, make_paper("p1", "An abstract."))

    dated_path = agent_pick.write_agent_pick_digest(conn, tmp_path, [
        AgentPickDecision(arxiv_id="p1", status="picked", reasoning="x", decided_at="t"),
    ])
    today = date.today().isoformat()
    assert dated_path == tmp_path / f"agent-pick-{today}.md"
    assert dated_path.exists()
    latest_path = tmp_path / "agent-pick-latest.md"
    assert latest_path.exists()
    assert latest_path.read_text() == dated_path.read_text()
