# arXiv Curator — Design Spec

Date: 2026-07-07

## Problem / Motivation

Keeping up with new arXiv papers is manual and low-signal: you either skim category feeds yourself or miss things. This project builds a personal backend/CLI that fetches newly published arXiv papers, summarizes them, ranks them against a stated interest profile, learns from your feedback (including how much of a paper you actually read), explains why each paper was ranked the way it was, and can be evaluated for ranking quality over time. The result should also be legible on GitHub — a visitor should be able to see curated output without running anything locally.

## Goals

- Fetch newest papers from arXiv, deduped, filtered by category.
- Summarize abstracts via a pluggable LLM provider (Gemini first, Anthropic as an alternative, local/offline fallback for CI).
- Rank papers against a free-text interest profile using embeddings + a feedback-adjusted score.
- Record feedback: explicit up/down rating, free-text notes, and self-reported read depth (pages read / total pages).
- Produce a grounded, human-readable "why this matches you" explanation per paper.
- Run ranking-quality evals against accumulated feedback.
- Publish digests as committed markdown (source of truth) and a generated static HTML site via GitHub Pages.
- Ship CI that runs without any API key (local provider), plus a scheduled workflow that runs the real pipeline with a provider key.

## Non-goals (v1)

- No PDF viewer / reader — read-depth is self-reported via CLI flag, not measured automatically.
- No email delivery of digests — viewing is via committed markdown and/or GitHub Pages.
- No multi-user support, no hosted database — single local SQLite file, single interest profile.
- No training of a real ML classifier — ranking adjustment uses a Rocchio-style centroid method, not a trained model.

## Architecture

```
arxiv-curator/
  README.md
  pyproject.toml
  .env.example
  .gitignore
  interests.yaml                 # editable interest profile (summary/topics/keywords/liked_examples)
  src/arxiv_curator/
    cli.py                       # Typer app — entry point
    config.py                    # env/config loading, default categories, paths
    models.py                    # Paper, Summary, Score, Feedback dataclasses
    db.py                        # SQLite schema + CRUD
    fetch.py                     # arXiv Atom API client + dedup
    llm/
      base.py                    # Protocols: Summarizer, Explainer
      anthropic_provider.py      # Claude-backed summarizer/explainer
      gemini_provider.py         # Gemini-backed summarizer/explainer
      local_provider.py          # offline fallback (extractive summary, template explanation)
      embeddings.py              # sentence-transformers embedder (always local, provider-independent)
      factory.py                 # picks provider from env/config
    rank.py                      # Rocchio-style centroid ranking + "why this matches"
    feedback.py                  # record/read feedback (rating, note, read-depth)
    digest.py                    # renders markdown digest + generates docs/ HTML mirror
    eval.py                      # ranking-quality eval (leave-one-out precision/NDCG vs baselines)
  tests/
    test_fetch.py
    test_db.py
    test_rank.py
    test_eval.py
  digests/                       # committed markdown output (source of truth)
  docs/                          # generated static HTML site, served by GitHub Pages
  .github/workflows/
    ci.yml                       # pytest on push/PR, local provider, no secrets needed
    daily-digest.yml             # scheduled fetch→summarize→rank→digest, commits result
```

## Components

**models.py** — plain dataclasses: `Paper` (arxiv_id, title, authors, abstract, categories, published, url), `Summary`, `Score` (similarity, feedback_adjustment, final_score, explanation), `Feedback` (arxiv_id, rating: up/down/none, pages_read, total_pages, note, created_at). Rating and read-depth are independent — either, both, or neither (with a note) can be logged.

**db.py** — SQLite at `data/arxiv_curator.db` (gitignored). Tables: `papers`, `summaries`, `scores`, `feedback`. Plain CRUD functions, no ORM — keeps the dependency footprint small and the data directly inspectable with the `sqlite3` CLI.

**fetch.py** — queries `export.arxiv.org/api/query` (Atom feed) via `requests` + `feedparser`, sorted by submission date, filtered by configured categories. Dedupes against the `papers` table by `arxiv_id` before inserting new rows.

**interests.yaml** — free text: `summary` (paragraph), `topics` (list), `keywords` (list), optional `liked_examples` (arxiv ids/blurbs). The whole document is embedded once into an "interest vector," recomputed whenever the file changes (hash-checked).

**llm/embeddings.py** — embeddings always run locally via `sentence-transformers` (`all-MiniLM-L6-v2`), independent of which LLM provider is configured for text generation. Neither Anthropic nor Gemini's chat APIs are used for embeddings; this keeps ranking free, fast, and deterministic, and avoids provider lock-in for the similarity math the whole system depends on.

**llm/base.py + factory.py** — `Summarizer.summarize(paper) -> str` and `Explainer.explain(paper, interest_profile, signals) -> str` protocols. `factory.get_provider()` selects a provider via `LLM_PROVIDER` env var (`gemini` / `anthropic` / `local`), defaulting by whichever API key is present (`GEMINI_API_KEY` or `ANTHROPIC_API_KEY`), falling back to the local/offline provider if neither is set. The local provider is deterministic (extractive summary: first N sentences of the abstract; templated explanation from the same signals used elsewhere) so tests and CI need no network access or secrets.

**rank.py** — Rocchio-style relevance feedback:
- `base_score = cosine(interest_vector, paper_vector)`
- `mean_liked` / `mean_disliked` = weighted centroids of embeddings of rated papers
- each feedback item's weight toward its centroid is `0.5 + 0.5 * (pages_read / total_pages)` when read-depth is present (defaulting to weight `1.0` when only a rating is given), so a paper read in full counts more than one barely opened
- `final_score = base_score + α · cosine(paper, mean_liked) − β · cosine(paper, mean_disliked)`, with α/β as tunable config constants (default 0.3 each)
- "Why this matches": deterministic signals — overlapping keywords/topics between abstract and interest profile, and the most-similar previously liked paper by cosine distance — are computed first, then passed to the `Explainer` LLM call so its explanation is grounded in real signal rather than free-floating justification.

**feedback.py** — `record_feedback(arxiv_id, rating=None, note=None, pages_read=None, total_pages=None)` and `list_feedback()`. No PDF viewer in v1: you read the paper in whatever tool you already use and self-report progress via `arxiv-curator feedback <id> --pages-read 5 --total-pages 12`.

**digest.py** — renders `digests/YYYY-MM-DD.md` (ranked list, summary, score, explanation, feedback command hints) as the committed source of truth, and regenerates a static HTML mirror into `docs/` (`docs/index.html` = latest digest, `docs/archive/YYYY-MM-DD.html` for history) using the `markdown` package plus a minimal shared template — no JS framework.

**eval.py** — leave-one-out evaluation: for each feedback item, rebuild the liked/disliked centroids from all *other* feedback, rank a snapshot of candidate papers, and measure where the held-out item lands. Reports Precision@5, Precision@10, NDCG@10, and MRR of liked papers, against two baselines (random order; similarity-only with no feedback adjustment). Reports "insufficient data" rather than a misleading number when feedback volume is too low (threshold: fewer than 5 rated papers).

## CLI surface

```
arxiv-curator fetch      --categories cs.AI,cs.LG,cs.CL,stat.ML --max-results 100
arxiv-curator summarize  [--limit N]
arxiv-curator rank       [--top N]
arxiv-curator show       <arxiv_id>
arxiv-curator feedback   <arxiv_id> [--rating up|down] [--note "..."] [--pages-read N --total-pages M]
arxiv-curator digest     [--top N] [--out digests/YYYY-MM-DD.md]
arxiv-curator eval
arxiv-curator run        # fetch + summarize + rank + digest, used by the scheduled workflow
```

## Data flow

1. `fetch` pulls new papers → `papers` table.
2. `summarize` generates a `Summary` per paper missing one, via the configured LLM provider.
3. `rank` embeds any un-embedded papers, computes Rocchio-adjusted scores + explanations → `scores` table.
4. `feedback` records a rating/read-depth/note against an `arxiv_id` → `feedback` table, which feeds back into the next `rank` run's centroids.
5. `digest` reads the latest `scores` + `summaries` and renders markdown + HTML.
6. `eval` reads `feedback` + historical embeddings to score ranking quality.

## Error handling

- Network failures in `fetch` (arXiv API) are caught and reported per-request; partial results already fetched are still stored (no all-or-nothing transaction spanning the whole fetch).
- LLM provider failures (rate limit, timeout, missing key) fall back to the local provider for that run rather than crashing the whole pipeline, with a warning printed to stderr.
- `rank`/`eval` with zero feedback rows degrade gracefully: ranking uses similarity-only (no Rocchio adjustment), and eval reports "insufficient data."
- Malformed `interests.yaml` (missing required `summary` field) fails fast with a clear CLI error before any network calls.

## Testing

- `test_fetch.py` — parsing of a mocked Atom feed response, dedup logic against pre-seeded `papers` rows.
- `test_db.py` — CRUD round-trips on a temporary SQLite file for all four tables.
- `test_rank.py` — cosine similarity and Rocchio centroid math against fixed, hand-computed dummy vectors (no real embedding model invoked).
- `test_eval.py` — precision/NDCG/MRR computed correctly against a small synthetic labeled dataset with known expected metrics.
- CI (`ci.yml`) runs all of the above using the local LLM provider — no API key required, fully reproducible.

## Open items deferred to later versions

- PDF viewer with automatic read tracking.
- Emailed digests.
- Trained (non-heuristic) ranking model, once enough feedback accumulates to make one worthwhile.
