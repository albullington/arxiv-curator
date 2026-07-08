from dataclasses import dataclass
from typing import Optional


@dataclass
class Paper:
    arxiv_id: str
    title: str
    authors: str
    abstract: str
    categories: str
    published: str
    url: str


@dataclass
class Summary:
    arxiv_id: str
    text: str
    created_at: str


@dataclass
class Score:
    arxiv_id: str
    similarity: float
    feedback_adjustment: float
    final_score: float
    explanation: str
    created_at: str


@dataclass
class Feedback:
    arxiv_id: str
    created_at: str
    rating: Optional[str] = None
    pages_read: Optional[int] = None
    total_pages: Optional[int] = None
    note: Optional[str] = None
