from datetime import datetime, timezone
from typing import Optional

import numpy as np

from arxiv_curator import db
from arxiv_curator.interests import load_interest_profile, profile_to_text
from arxiv_curator.llm.embeddings import cosine_similarity, embed_texts
from arxiv_curator.models import Feedback, Score

DEFAULT_ALPHA = 0.3
DEFAULT_BETA = 0.3


def get_paper_vectors(conn, papers, client) -> dict:
    arxiv_ids = [p.arxiv_id for p in papers]
    vectors_by_id = db.get_embeddings(conn, arxiv_ids)
    missing = [p for p in papers if p.arxiv_id not in vectors_by_id]
    if missing:
        new_vectors = embed_texts([p.abstract for p in missing], client)
        for paper, vector in zip(missing, new_vectors):
            db.upsert_embedding(conn, paper.arxiv_id, vector)
            vectors_by_id[paper.arxiv_id] = vector
    return vectors_by_id


def feedback_weight(feedback: Feedback) -> float:
    if feedback.pages_read is not None and feedback.total_pages:
        return 0.5 + 0.5 * (feedback.pages_read / feedback.total_pages)
    return 1.0


def compute_centroids(
    feedback_items: list[Feedback], vectors_by_id: dict[str, np.ndarray]
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    liked_vectors = []
    disliked_vectors = []
    for fb in feedback_items:
        vec = vectors_by_id.get(fb.arxiv_id)
        if vec is None:
            continue
        weight = feedback_weight(fb)
        if fb.rating == "up":
            liked_vectors.append(vec * weight)
        elif fb.rating == "down":
            disliked_vectors.append(vec * weight)
    mean_liked = np.mean(liked_vectors, axis=0) if liked_vectors else None
    mean_disliked = np.mean(disliked_vectors, axis=0) if disliked_vectors else None
    return mean_liked, mean_disliked


def score_paper(
    paper_vector: np.ndarray,
    interest_vector: np.ndarray,
    mean_liked: Optional[np.ndarray],
    mean_disliked: Optional[np.ndarray],
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
) -> tuple[float, float, float]:
    similarity = cosine_similarity(paper_vector, interest_vector)
    adjustment = 0.0
    if mean_liked is not None:
        adjustment += alpha * cosine_similarity(paper_vector, mean_liked)
    if mean_disliked is not None:
        adjustment -= beta * cosine_similarity(paper_vector, mean_disliked)
    final = similarity + adjustment
    return similarity, adjustment, final


def overlapping_keywords(abstract: str, keywords: list[str]) -> list[str]:
    abstract_lower = abstract.lower()
    return [kw for kw in keywords if kw.lower() in abstract_lower]


def most_similar_liked(
    paper_vector: np.ndarray, liked_vectors_by_id: dict[str, np.ndarray]
) -> Optional[str]:
    best_id = None
    best_score = -1.0
    for arxiv_id, vec in liked_vectors_by_id.items():
        sim = cosine_similarity(paper_vector, vec)
        if sim > best_score:
            best_id, best_score = arxiv_id, sim
    return best_id


def rank_papers(conn, interests_path, client) -> list[Score]:
    """Score every paper against the interest profile and feedback centroid.

    Does not generate "why this matches" explanations -- those are expensive
    LLM calls that only matter for papers a caller actually intends to show
    (e.g. the digest's top N). Use explain_papers for that, once the set of
    papers worth explaining is known.
    """
    profile = load_interest_profile(interests_path)
    profile_text = profile_to_text(profile)
    interest_vector = embed_texts([profile_text], client)[0]

    papers = db.list_papers(conn)
    if not papers:
        return []
    vectors_by_id = get_paper_vectors(conn, papers, client)

    feedback_items = db.list_feedback(conn)
    mean_liked, mean_disliked = compute_centroids(feedback_items, vectors_by_id)

    results = []
    for paper in papers:
        vec = vectors_by_id[paper.arxiv_id]
        similarity, adjustment, final = score_paper(vec, interest_vector, mean_liked, mean_disliked)
        existing_score = db.get_score(conn, paper.arxiv_id)
        explanation = existing_score.explanation if existing_score is not None else ""
        score = Score(
            arxiv_id=paper.arxiv_id, similarity=similarity, feedback_adjustment=adjustment,
            final_score=final, explanation=explanation,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        db.upsert_score(conn, score)
        results.append(score)
    return results


def explain_papers(conn, interests_path, provider, client, arxiv_ids: list[str]) -> None:
    """Generate and persist "why this matches" explanations for already-scored
    papers, skipping any that already have one. Requires rank_papers to have
    run first so a score exists for each id.
    """
    pending_ids = [
        aid for aid in arxiv_ids
        if (existing := db.get_score(conn, aid)) is not None and not existing.explanation
    ]
    if not pending_ids:
        return

    profile = load_interest_profile(interests_path)
    profile_text = profile_to_text(profile)

    papers_by_id = {p.arxiv_id: p for p in db.list_papers(conn) if p.arxiv_id in pending_ids}
    vectors_by_id = get_paper_vectors(conn, list(papers_by_id.values()), client)

    feedback_items = db.list_feedback(conn)
    liked_ids = [fb.arxiv_id for fb in feedback_items if fb.rating == "up"]
    liked_vectors_by_id = db.get_embeddings(conn, liked_ids)

    for arxiv_id in pending_ids:
        paper = papers_by_id[arxiv_id]
        vec = vectors_by_id[arxiv_id]
        keywords = overlapping_keywords(paper.abstract, profile.keywords)
        closest = most_similar_liked(vec, liked_vectors_by_id)
        signals = {"overlapping_keywords": keywords, "most_similar_liked": closest}
        explanation = provider.explain(paper, profile_text, signals)

        existing = db.get_score(conn, arxiv_id)
        db.upsert_score(conn, Score(
            arxiv_id=existing.arxiv_id, similarity=existing.similarity,
            feedback_adjustment=existing.feedback_adjustment, final_score=existing.final_score,
            explanation=explanation, created_at=existing.created_at,
        ))
