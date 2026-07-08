from arxiv_curator.models import Paper, Summary, Score, Feedback


def test_paper_fields():
    paper = Paper(
        arxiv_id="2601.00001",
        title="A Paper",
        authors="A. Author",
        abstract="An abstract.",
        categories="cs.AI",
        published="2026-01-01T00:00:00Z",
        url="https://arxiv.org/abs/2601.00001",
    )
    assert paper.arxiv_id == "2601.00001"


def test_feedback_defaults_to_none():
    feedback = Feedback(arxiv_id="2601.00001", created_at="2026-01-01T00:00:00Z")
    assert feedback.rating is None
    assert feedback.pages_read is None
    assert feedback.total_pages is None
    assert feedback.note is None


def test_feedback_with_rating_and_read_depth():
    feedback = Feedback(
        arxiv_id="2601.00001",
        created_at="2026-01-01T00:00:00Z",
        rating="up",
        pages_read=5,
        total_pages=12,
        note="good background section",
    )
    assert feedback.rating == "up"
    assert feedback.pages_read == 5
