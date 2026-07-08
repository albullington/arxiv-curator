import random

import numpy as np

from arxiv_curator import db, rank
from arxiv_curator.interests import load_interest_profile, profile_to_text
from arxiv_curator.llm.embeddings import embed_texts

MIN_FEEDBACK_FOR_EVAL = 5


def precision_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    top_k = ranked_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for aid in top_k if aid in relevant_ids)
    return hits / len(top_k)


def _dcg(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    return sum(
        1.0 / np.log2(i + 2) for i, aid in enumerate(ranked_ids[:k]) if aid in relevant_ids
    )


def ndcg_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    actual = _dcg(ranked_ids, relevant_ids, k)
    ideal_order = [aid for aid in ranked_ids if aid in relevant_ids] + \
        [aid for aid in ranked_ids if aid not in relevant_ids]
    ideal = _dcg(ideal_order, relevant_ids, k)
    return actual / ideal if ideal > 0 else 0.0


def mrr(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    for i, aid in enumerate(ranked_ids):
        if aid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate(
    feedback_items: list,
    vectors_by_id: dict[str, np.ndarray],
    interest_vector: np.ndarray,
    rng_seed: int = 42,
) -> dict:
    rated = [f for f in feedback_items if f.rating in ("up", "down")]
    if len(rated) < MIN_FEEDBACK_FOR_EVAL:
        return {"status": "insufficient_data", "rated_count": len(rated)}

    candidate_ids = list(vectors_by_id.keys())
    rng = random.Random(rng_seed)

    metrics = {
        "feedback_adjusted": {"precision_at_5": [], "precision_at_10": [], "ndcg_at_10": [], "mrr": []},
        "similarity_only_baseline": {"precision_at_5": [], "ndcg_at_10": [], "mrr": []},
        "random_baseline": {"precision_at_5": [], "ndcg_at_10": [], "mrr": []},
    }

    n_evaluated = 0
    for held_out in rated:
        if held_out.rating != "up":
            continue
        relevant = {held_out.arxiv_id}
        other_feedback = [f for f in rated if f.arxiv_id != held_out.arxiv_id]
        mean_liked, mean_disliked = rank.compute_centroids(other_feedback, vectors_by_id)

        adjusted_scored, similarity_scored = [], []
        for aid in candidate_ids:
            vec = vectors_by_id[aid]
            similarity, _, final = rank.score_paper(vec, interest_vector, mean_liked, mean_disliked)
            adjusted_scored.append((aid, final))
            similarity_scored.append((aid, similarity))

        adjusted_ranked = [aid for aid, _ in sorted(adjusted_scored, key=lambda x: x[1], reverse=True)]
        similarity_ranked = [aid for aid, _ in sorted(similarity_scored, key=lambda x: x[1], reverse=True)]
        random_ranked = candidate_ids[:]
        rng.shuffle(random_ranked)

        metrics["feedback_adjusted"]["precision_at_5"].append(precision_at_k(adjusted_ranked, relevant, 5))
        metrics["feedback_adjusted"]["precision_at_10"].append(precision_at_k(adjusted_ranked, relevant, 10))
        metrics["feedback_adjusted"]["ndcg_at_10"].append(ndcg_at_k(adjusted_ranked, relevant, 10))
        metrics["feedback_adjusted"]["mrr"].append(mrr(adjusted_ranked, relevant))

        metrics["similarity_only_baseline"]["precision_at_5"].append(precision_at_k(similarity_ranked, relevant, 5))
        metrics["similarity_only_baseline"]["ndcg_at_10"].append(ndcg_at_k(similarity_ranked, relevant, 10))
        metrics["similarity_only_baseline"]["mrr"].append(mrr(similarity_ranked, relevant))

        metrics["random_baseline"]["precision_at_5"].append(precision_at_k(random_ranked, relevant, 5))
        metrics["random_baseline"]["ndcg_at_10"].append(ndcg_at_k(random_ranked, relevant, 10))
        metrics["random_baseline"]["mrr"].append(mrr(random_ranked, relevant))
        n_evaluated += 1

    return {
        "status": "ok",
        "n_evaluated": n_evaluated,
        "feedback_adjusted": {k: _avg(v) for k, v in metrics["feedback_adjusted"].items()},
        "similarity_only_baseline": {k: _avg(v) for k, v in metrics["similarity_only_baseline"].items()},
        "random_baseline": {k: _avg(v) for k, v in metrics["random_baseline"].items()},
    }


def run_eval(conn, interests_path, client) -> dict:
    profile = load_interest_profile(interests_path)
    interest_vector = embed_texts([profile_to_text(profile)], client)[0]
    papers = db.list_papers(conn)
    vectors = embed_texts([p.abstract for p in papers], client) if papers else np.empty((0, 0))
    vectors_by_id = {p.arxiv_id: vec for p, vec in zip(papers, vectors)}
    feedback_items = db.list_feedback(conn)
    return evaluate(feedback_items, vectors_by_id, interest_vector)
