# arXiv Curator

Fetches newest arXiv papers, summarizes them, ranks them against your
interests, learns from your feedback (including how much you actually
read), explains why each paper matched, and evaluates ranking quality.

## Setup

    python -m venv .venv
    source .venv/bin/activate
    pip install -e ".[dev]"
    cp .env.example .env   # add your GEMINI_API_KEY
    git clone <your-arxiv-curator-data repo URL> ~/arxiv-curator-data

Edit `interests.yaml` to describe what you actually care about.

By default the database lives in `~/arxiv-curator-data/arxiv_curator.db`,
a clone of a separate private repo dedicated to your papers, ratings,
and notes (kept out of this public repo). Override the location with
`ARXIV_CURATOR_DATA_DIR` in `.env`. Run `arxiv-curator sync` before a
session to pull down anything the daily CI run added, and after a
session to push your local adds/ratings so tomorrow's run sees them.

## Usage

    arxiv-curator fetch --categories cs.AI,cs.LG,cs.CL,stat.ML
    arxiv-curator add <arxiv_id>
    arxiv-curator summarize
    arxiv-curator rank
    arxiv-curator show <arxiv_id>
    arxiv-curator feedback <arxiv_id> --rating up
    arxiv-curator feedback <arxiv_id> --pages-read 5 --total-pages 12
    arxiv-curator digest
    arxiv-curator eval
    arxiv-curator run    # fetch + summarize + rank + digest in one shot
    arxiv-curator sync   # pull remote changes, push local changes
    arxiv-curator agent-pick   # confidence-gated agentic alternative to `digest`

## How ranking works

Papers are embedded (Gemini embeddings) and scored by cosine similarity
to your `interests.yaml` profile. As you rate papers up/down (optionally
with how much you actually read), a Rocchio-style centroid of liked and
disliked papers nudges future scores toward what you've responded to.
Each ranked paper gets a grounded "why this matches" explanation, built
from real signals (overlapping keywords, closest liked paper) rather
than free-floating LLM justification.
Each paper's explanation is generated once (the first time it's ranked)
and reused after that, even as your interests or feedback evolve --
only the numeric score recomputes on every run. This keeps cost and
runtime bounded by how many papers are new, not by how many you've
accumulated.

## Viewing digests

`digests/latest.md` always has the most recent run; `digests/YYYY-MM-DD.md`
keeps history. Both render natively when browsing this repo on GitHub.

`arxiv-curator agent-pick` writes its own parallel output --
`digests/agent-pick-latest.md` and `digests/agent-pick-YYYY-MM-DD.md` --
instead of touching the regular digest files, so the two can be compared
side by side. It reasons over a shortlist of candidates with an LLM
tool-calling loop against a career/learning-value bar (deepens
understanding of LLMs and their infrastructure, or suggests something
concretely triable), surfacing up to 3 picks per run and holding
borderline papers for future reconsideration instead of forcing a
verdict. Scheduled weekly (Fridays) via `.github/workflows/agent-pick.yml`,
alongside the existing daily digest -- an experiment to run in parallel,
not a replacement.

## Evals

`arxiv-curator eval` runs a leave-one-out evaluation against your stored
feedback (Precision@5/@10, NDCG@10, MRR), compared against a
similarity-only baseline and a random baseline. Needs at least 5 rated
papers to produce real numbers.

## Tests

    pytest
