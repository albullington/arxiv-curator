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


VALID_STATUSES = {"picked", "held", "rejected"}
MAX_PICKS_PER_RUN = 3


class InvalidFinalizePayload(Exception):
    pass


def get_paper_detail(conn, arxiv_id: str) -> dict:
    paper = db.get_paper(conn, arxiv_id)
    if paper is None:
        return {"error": f"No such paper: {arxiv_id}"}
    summary = db.get_summary(conn, arxiv_id)
    return {
        "arxiv_id": paper.arxiv_id,
        "title": paper.title,
        "authors": paper.authors,
        "abstract": paper.abstract,
        "categories": paper.categories,
        "summary": summary.text if summary else None,
    }


def get_feedback_history(conn, client, arxiv_id: str, top_k: int = 3) -> dict:
    candidate_vec = db.get_embeddings(conn, [arxiv_id]).get(arxiv_id)
    if candidate_vec is None:
        return {"similar_rated_papers": []}

    rated_feedback = [f for f in db.list_feedback(conn) if f.rating in ("up", "down")]
    rated_ids = list({f.arxiv_id for f in rated_feedback})
    vectors_by_id = db.get_embeddings(conn, rated_ids)

    scored = []
    for fb in rated_feedback:
        if fb.arxiv_id == arxiv_id:
            continue
        vec = vectors_by_id.get(fb.arxiv_id)
        if vec is None:
            continue
        scored.append((cosine_similarity(candidate_vec, vec), fb))
    scored.sort(key=lambda pair: pair[0], reverse=True)

    return {
        "similar_rated_papers": [
            {
                "arxiv_id": fb.arxiv_id, "rating": fb.rating, "note": fb.note,
                "similarity": round(sim, 4),
            }
            for sim, fb in scored[:top_k]
        ]
    }


def validate_decisions(shortlist_ids: set, raw_decisions: list) -> list:
    seen_ids = set()
    picked_count = 0
    for entry in raw_decisions:
        arxiv_id = entry.get("arxiv_id")
        status = entry.get("status")
        reasoning = entry.get("reasoning")
        if arxiv_id not in shortlist_ids:
            raise InvalidFinalizePayload(f"{arxiv_id!r} is not in the shortlist")
        if status not in VALID_STATUSES:
            raise InvalidFinalizePayload(f"invalid status {status!r} for {arxiv_id}")
        if not reasoning:
            raise InvalidFinalizePayload(f"missing reasoning for {arxiv_id}")
        if arxiv_id in seen_ids:
            raise InvalidFinalizePayload(f"duplicate decision for {arxiv_id}")
        seen_ids.add(arxiv_id)
        if status == "picked":
            picked_count += 1

    missing = shortlist_ids - seen_ids
    if missing:
        raise InvalidFinalizePayload(f"missing decisions for {sorted(missing)}")
    if picked_count > MAX_PICKS_PER_RUN:
        raise InvalidFinalizePayload(f"{picked_count} papers picked, max is {MAX_PICKS_PER_RUN}")

    return raw_decisions
