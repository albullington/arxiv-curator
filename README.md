# arXiv Curator

Fetches newest arXiv papers, summarizes them, ranks them against your
interests, learns from your feedback (including how much you actually
read), explains why each paper matched, and evaluates ranking quality.

## Setup

    python -m venv .venv
    source .venv/bin/activate
    pip install -e ".[dev]"
    cp .env.example .env   # add your GEMINI_API_KEY

Edit `interests.yaml` to describe what you actually care about.

## Usage

    arxiv-curator fetch --categories cs.AI,cs.LG,cs.CL,stat.ML
    arxiv-curator summarize
    arxiv-curator rank
    arxiv-curator show <arxiv_id>
    arxiv-curator feedback <arxiv_id> --rating up
    arxiv-curator feedback <arxiv_id> --pages-read 5 --total-pages 12
    arxiv-curator digest
    arxiv-curator eval
    arxiv-curator run   # fetch + summarize + rank + digest in one shot

## How ranking works

Papers are embedded (Gemini embeddings) and scored by cosine similarity
to your `interests.yaml` profile. As you rate papers up/down (optionally
with how much you actually read), a Rocchio-style centroid of liked and
disliked papers nudges future scores toward what you've responded to.
Each ranked paper gets a grounded "why this matches" explanation, built
from real signals (overlapping keywords, closest liked paper) rather
than free-floating LLM justification.

## Viewing digests

`digests/latest.md` always has the most recent run; `digests/YYYY-MM-DD.md`
keeps history. Both render natively when browsing this repo on GitHub.

## Evals

`arxiv-curator eval` runs a leave-one-out evaluation against your stored
feedback (Precision@5/@10, NDCG@10, MRR), compared against a
similarity-only baseline and a random baseline. Needs at least 5 rated
papers to produce real numbers.

## Tests

    pytest
