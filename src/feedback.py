from datetime import datetime, timezone
from typing import Optional

from arxiv_curator import db
from arxiv_curator.models import Feedback

VALID_RATINGS = {"up", "down"}


def record_feedback(
    conn,
    arxiv_id: str,
    rating: Optional[str] = None,
    note: Optional[str] = None,
    pages_read: Optional[int] = None,
    total_pages: Optional[int] = None,
) -> None:
    if not db.paper_exists(conn, arxiv_id):
        raise ValueError(f"No such paper: {arxiv_id}")
    if rating is None and note is None and pages_read is None:
        raise ValueError("Must provide at least one of rating, note, or pages_read")
    if rating is not None and rating not in VALID_RATINGS:
        raise ValueError(f"rating must be one of {VALID_RATINGS}, got {rating!r}")
    if pages_read is not None and total_pages is not None and pages_read > total_pages:
        raise ValueError("pages_read cannot exceed total_pages")

    db.insert_feedback(conn, Feedback(
        arxiv_id=arxiv_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        rating=rating,
        pages_read=pages_read,
        total_pages=total_pages,
        note=note,
    ))
