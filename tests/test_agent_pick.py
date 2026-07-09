import numpy as np

from arxiv_curator import agent_pick, db, rank
from arxiv_curator.models import AgentPickDecision, Paper


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
