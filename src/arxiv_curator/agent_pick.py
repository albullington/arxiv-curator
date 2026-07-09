from datetime import date, datetime, timezone
from pathlib import Path

from arxiv_curator import db
from arxiv_curator.llm.agent_loop import ToolSpec, run_tool_loop
from arxiv_curator.llm.embeddings import cosine_similarity, embed_texts
from arxiv_curator.models import AgentPickDecision
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


GET_PAPER_DETAIL_SCHEMA = {
    "type": "object",
    "properties": {"arxiv_id": {"type": "string"}},
    "required": ["arxiv_id"],
}

GET_FEEDBACK_HISTORY_SCHEMA = {
    "type": "object",
    "properties": {"arxiv_id": {"type": "string"}},
    "required": ["arxiv_id"],
}

FINALIZE_SCHEMA = {
    "type": "object",
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "arxiv_id": {"type": "string"},
                    "status": {"type": "string", "enum": sorted(VALID_STATUSES)},
                    "reasoning": {"type": "string"},
                },
                "required": ["arxiv_id", "status", "reasoning"],
            },
        },
    },
    "required": ["decisions"],
}


def build_system_prompt(criteria_text: str, shortlist: list, held_reasoning_by_id: dict) -> str:
    lines = [
        "You are picking up to 3 arXiv papers per run against this bar:",
        criteria_text,
        "",
        "Candidates:",
    ]
    for paper in shortlist:
        lines.append(f"- {paper.arxiv_id}: {paper.title}")
        lines.append(f"  Abstract: {paper.abstract}")
        if paper.arxiv_id in held_reasoning_by_id:
            lines.append(f"  Previously held because: {held_reasoning_by_id[paper.arxiv_id]}")
    lines.append("")
    lines.append(
        "Use get_paper_detail and get_feedback_history if you need more than "
        "the abstract to judge a candidate. When ready, call finalize with a "
        "decision (picked/held/rejected) and reasoning for every candidate "
        "listed above -- up to 3 may be picked."
    )
    return "\n".join(lines)


def build_tools(conn, client) -> list:
    return [
        ToolSpec(
            name="get_paper_detail",
            description="Get the full abstract, cached summary, authors, and categories for a candidate paper.",
            parameters_json_schema=GET_PAPER_DETAIL_SCHEMA,
            handler=lambda arxiv_id: get_paper_detail(conn, arxiv_id),
        ),
        ToolSpec(
            name="get_feedback_history",
            description="Get the ~3 most similar previously-rated papers to a candidate, with rating and note.",
            parameters_json_schema=GET_FEEDBACK_HISTORY_SCHEMA,
            handler=lambda arxiv_id: get_feedback_history(conn, client, arxiv_id),
        ),
        ToolSpec(
            name="finalize",
            description="Submit final decisions for every candidate.",
            parameters_json_schema=FINALIZE_SCHEMA,
            handler=None,
        ),
    ]


def run_agent_pick(conn, client, criteria_text: str = CRITERIA_TEXT) -> list:
    shortlist, held_reasoning_by_id = build_shortlist(conn, client, criteria_text)
    if not shortlist:
        return []

    system_prompt = build_system_prompt(criteria_text, shortlist, held_reasoning_by_id)
    tools = build_tools(conn, client)
    result = run_tool_loop(
        client, system_prompt, tools,
        finalize_tool_name="finalize", max_tool_calls=AGENT_PICK_MAX_TOOL_CALLS,
    )

    shortlist_ids = {p.arxiv_id for p in shortlist}
    decisions = validate_decisions(shortlist_ids, result.get("decisions", []))

    now = datetime.now(timezone.utc).isoformat()
    persisted = []
    for entry in decisions:
        record = AgentPickDecision(
            arxiv_id=entry["arxiv_id"], status=entry["status"],
            reasoning=entry["reasoning"], decided_at=now,
        )
        db.upsert_agent_pick_decision(conn, record)
        persisted.append(record)
    return persisted


def render_agent_pick_digest(conn, picked: list) -> str:
    lines = [f"# Agent Pick -- {date.today().isoformat()}", ""]
    rendered_any = False
    for decision in picked:
        paper = db.get_paper(conn, decision.arxiv_id)
        if paper is None:
            continue
        rendered_any = True
        summary = db.get_summary(conn, decision.arxiv_id)
        lines.append(f"## [{paper.title}]({paper.url})")
        lines.append(f"**arXiv:** {paper.arxiv_id}")
        lines.append("")
        lines.append(summary.text if summary else paper.abstract)
        lines.append("")
        lines.append(f"**Why this was picked:** {decision.reasoning}")
        lines.append("")
        lines.append(f"`arxiv-curator feedback {paper.arxiv_id} --rating up`")
        lines.append("")
    if not rendered_any:
        lines.append("Nothing cleared the bar this run.")
        lines.append("")
    return "\n".join(lines)


def write_agent_pick_digest(conn, out_dir, picked: list) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    content = render_agent_pick_digest(conn, picked)
    dated_path = out_dir / f"agent-pick-{date.today().isoformat()}.md"
    dated_path.write_text(content)
    (out_dir / "agent-pick-latest.md").write_text(content)
    return dated_path
