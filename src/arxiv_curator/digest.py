from datetime import date
from pathlib import Path
from typing import Optional

from arxiv_curator import db
from arxiv_curator.models import Score


def select_digest_scores(conn, top_n: int = 10, since: Optional[str] = None) -> list[Score]:
    if since is not None:
        eligible_ids = {p.arxiv_id for p in db.list_papers_since(conn, since)}
        scores = [s for s in db.list_scores(conn) if s.arxiv_id in eligible_ids]
    else:
        scores = db.list_scores(conn)
    return sorted(scores, key=lambda s: s.final_score, reverse=True)[:top_n]


def render_digest(conn, top_n: int = 10, since: Optional[str] = None) -> str:
    scores = select_digest_scores(conn, top_n, since)
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


def write_digest(conn, out_dir: Path, top_n: int = 10, since: Optional[str] = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    content = render_digest(conn, top_n, since)
    dated_path = out_dir / f"{date.today().isoformat()}.md"
    dated_path.write_text(content)
    (out_dir / "latest.md").write_text(content)
    return dated_path
