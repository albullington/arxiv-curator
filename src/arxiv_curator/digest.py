from datetime import date
from pathlib import Path

from arxiv_curator import db


def render_digest(conn, top_n: int = 20) -> str:
    scores = sorted(db.list_scores(conn), key=lambda s: s.final_score, reverse=True)[:top_n]
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


def write_digest(conn, out_dir: Path, top_n: int = 20) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    content = render_digest(conn, top_n)
    dated_path = out_dir / f"{date.today().isoformat()}.md"
    dated_path.write_text(content)
    (out_dir / "latest.md").write_text(content)
    return dated_path
