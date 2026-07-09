from arxiv_curator import db
from arxiv_curator.llm.embeddings import cosine_similarity, embed_texts
from arxiv_curator.rank import get_paper_vectors

AGENT_PICK_SHORTLIST_SIZE = 10
AGENT_PICK_MAX_TOOL_CALLS = 8

CRITERIA_TEXT = (
    "Papers that would deepen understanding of large language models and "
    "the infrastructure around them, or suggest something concretely "
    "triable in a sandbox project or at work -- not just papers that are "
    "topically similar to past interests."
)


def build_shortlist(conn, client, criteria_text: str = CRITERIA_TEXT, limit: int = AGENT_PICK_SHORTLIST_SIZE):
    new_papers = db.papers_without_agent_pick_decision(conn)
    held_decisions = db.list_held_agent_pick_decisions(conn)
    held_papers = [db.get_paper(conn, d.arxiv_id) for d in held_decisions]
    held_reasoning_by_id = {d.arxiv_id: d.reasoning for d in held_decisions}

    shortlisted_new = []
    if new_papers:
        criteria_vector = embed_texts([criteria_text], client)[0]
        vectors_by_id = get_paper_vectors(conn, new_papers, client)
        ranked = sorted(
            new_papers,
            key=lambda p: cosine_similarity(vectors_by_id[p.arxiv_id], criteria_vector),
            reverse=True,
        )
        shortlisted_new = ranked[:limit]

    return shortlisted_new + held_papers, held_reasoning_by_id
