# arXiv Curator — Design Spec

Date: 2026-07-07

## Problem / Motivation

Keeping up with new arXiv papers is manual and low-signal: you either skim category feeds yourself or miss things. This project builds a personal backend/CLI that fetches newly published arXiv papers, summarizes them, ranks them against a stated interest profile, learns from your feedback (including how much of a paper you actually read), explains why each paper was ranked the way it was, and can be evaluated for ranking quality over time. The result should also be legible on GitHub — a visitor should be able to see curated output without running anything locally.

## Goals

- Fetch newest papers from arXiv, deduped, filtered by category.
- Summarize abstracts via Gemini, behind a `Summarizer`/`Explainer` interface general enough to add another provider later.
- Rank papers against a free-text interest profile using embeddings + a feedback-adjusted score.
- Record feedback: explicit up/down rating, free-text notes, and self-reported read depth (pages read / total pages).
- Produce a grounded, human-readable "why this matches you" explanation per paper.
- Run ranking-quality evals against accumulated feedback.
- Publish digests as committed markdown — GitHub renders `.md` natively in the browser, so this alone satisfies "viewable on GitHub."
- Ship CI covering everything that doesn't require an LLM call (fetch parsing, db, rank math, eval math), plus a scheduled workflow that runs the real pipeline with a Gemini key.

## Non-goals (v1)

- No PDF viewer / reader — read-depth is self-reported via CLI flag, not measured automatically.
- No email delivery of digests, no GitHub Pages / generated HTML site — a committed markdown file is sufficient, since GitHub already renders it in the browser.
- No offline/local LLM fallback provider — Gemini is the only provider implemented; if it's unavailable, the affected command fails with a clear error rather than silently degrading to lower-quality output.
- No second LLM provider (e.g. Anthropic) implemented in v1 — the interface is provider-agnostic, but only Gemini is built.
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
      gemini_provider.py         # Gemini-backed summarizer/explainer (only implementation in v1)
      embeddings.py              # sentence-transformers embedder (always local, provider-independent)
      factory.py                 # reads GEMINI_API_KEY, constructs the Gemini provider
    rank.py                      # Rocchio-style centroid ranking + "why this matches"
    feedback.py                  # record/read feedback (rating, note, read-depth)
    digest.py                    # renders markdown digest (digests/YYYY-MM-DD.md + digests/latest.md)
    eval.py                      # ranking-quality eval (leave-one-out precision/NDCG vs baselines)
  tests/
    test_fetch.py
    test_db.py
    test_rank.py
    test_eval.py
  digests/                       # committed markdown output — this is "viewable on GitHub"
  .github/workflows/
    ci.yml                       # pytest on push/PR — no LLM calls in any test, no secrets needed
    daily-digest.yml             # scheduled fetch→summarize→rank→digest, commits digests/ only
```

**Scheduled workflow state:** the sqlite db (`data/arxiv_curator.db`) stays gitignored — it holds your feedback notes and ratings, which shouldn't be forced into a public repo's git history just to make the daily workflow work. `daily-digest.yml` persists it across scheduled runs using `actions/cache` (a mutable cache entry keyed by a fixed prefix, restored at the start of each run and re-saved at the end) rather than committing it. Only `digests/*.md` gets committed by the workflow.

## Components

**models.py** — plain dataclasses: `Paper` (arxiv_id, title, authors, abstract, categories, published, url), `Summary`, `Score` (similarity, feedback_adjustment, final_score, explanation), `Feedback` (arxiv_id, rating: up/down/none, pages_read, total_pages, note, created_at). Rating and read-depth are independent — either, both, or neither (with a note) can be logged.

**db.py** — SQLite at `data/arxiv_curator.db` (gitignored). Tables: `papers`, `summaries`, `scores`, `feedback`. Plain CRUD functions, no ORM — keeps the dependency footprint small and the data directly inspectable with the `sqlite3` CLI.

**fetch.py** — queries `export.arxiv.org/api/query` (Atom feed) via `requests` + `feedparser`, sorted by submission date, filtered by configured categories. Dedupes against the `papers` table by `arxiv_id` before inserting new rows.

**interests.yaml** — free text: `summary` (paragraph), `topics` (list), `keywords` (list), optional `liked_examples` (arxiv ids/blurbs). The whole document is embedded once into an "interest vector," recomputed whenever the file changes (hash-checked).

**llm/embeddings.py** — thin wrapper around Gemini's embeddings endpoint (`embed_texts(texts: list[str]) -> np.ndarray`). Since Gemini is the only provider in v1, using its embeddings API avoids pulling in a ~2GB local model (`sentence-transformers` + torch) for a capability the provider already offers. This means `rank`/`eval` need network + `GEMINI_API_KEY` to embed papers — the same requirement `summarize`/`explain` already have, so it's not a new constraint in practice. Pure ranking/eval math (`rank.py`'s scoring functions, `eval.py`'s metrics) is still tested against fixed, hand-computed vectors with no network call; only the thin orchestration layer that wires embeddings + db + LLM together needs `embed_texts` monkeypatched in tests.

**llm/base.py + factory.py** — `Summarizer.summarize(paper) -> str` and `Explainer.explain(paper, interest_profile, signals) -> str` protocols, implemented by `gemini_provider.py`. `factory.get_provider()` reads `GEMINI_API_KEY` and constructs the Gemini provider; if the key is missing, it raises a clear configuration error immediately rather than silently degrading. The protocol exists so a second provider can be added later without touching `rank.py`, `digest.py`, or the CLI — but no second provider ships in v1.

**rank.py** — Rocchio-style relevance feedback:
- `base_score = cosine(interest_vector, paper_vector)`
- `mean_liked` / `mean_disliked` = weighted centroids of embeddings of rated papers
- each feedback item's weight toward its centroid is `0.5 + 0.5 * (pages_read / total_pages)` when read-depth is present (defaulting to weight `1.0` when only a rating is given), so a paper read in full counts more than one barely opened
- `final_score = base_score + α · cosine(paper, mean_liked) − β · cosine(paper, mean_disliked)`, with α/β as tunable config constants (default 0.3 each)
- "Why this matches": deterministic signals — overlapping keywords/topics between abstract and interest profile, and the most-similar previously liked paper by cosine distance — are computed first, then passed to the `Explainer` LLM call so its explanation is grounded in real signal rather than free-floating justification.

**feedback.py** — `record_feedback(arxiv_id, rating=None, note=None, pages_read=None, total_pages=None)` and `list_feedback()`. No PDF viewer in v1: you read the paper in whatever tool you already use and self-report progress via `arxiv-curator feedback <id> --pages-read 5 --total-pages 12`.

**digest.py** — renders `digests/YYYY-MM-DD.md` (ranked list, summary, score, explanation, feedback command hints) as the committed source of truth, and overwrites `digests/latest.md` with the same content so there's one stable link to "today's digest" without needing to know the date. GitHub renders both natively in the browser — no HTML generation, no Pages setup.

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
5. `digest` reads the latest `scores` + `summaries` and renders `digests/YYYY-MM-DD.md` + `digests/latest.md`.
6. `eval` reads `feedback` + historical embeddings to score ranking quality.

## Error handling

- Network failures in `fetch` (arXiv API) are caught and reported per-request; partial results already fetched are still stored (no all-or-nothing transaction spanning the whole fetch).
- Missing `GEMINI_API_KEY` or a Gemini API error (rate limit, timeout) fails the `summarize`/`rank` (explanation step) command clearly and stops — no silent fallback, since there's no second provider to fall back to.
- `rank`/`eval` with zero feedback rows degrade gracefully: ranking uses similarity-only (no Rocchio adjustment), and eval reports "insufficient data."
- Malformed `interests.yaml` (missing required `summary` field) fails fast with a clear CLI error before any network calls.

## Testing

- `test_fetch.py` — parsing of a mocked Atom feed response, dedup logic against pre-seeded `papers` rows.
- `test_db.py` — CRUD round-trips on a temporary SQLite file for all four tables.
- `test_rank.py` — cosine similarity and Rocchio centroid math against fixed, hand-computed dummy vectors (no real embedding model invoked).
- `test_eval.py` — precision/NDCG/MRR computed correctly against a small synthetic labeled dataset with known expected metrics.
- None of the above tests call Gemini or need any API key — `summarize`/`rank`'s LLM-calling paths aren't unit tested against the real API in v1, only the pure logic around them (embedding math, signal computation). CI (`ci.yml`) runs the full suite with zero secrets.

## Open items deferred to later versions

- PDF viewer with automatic read tracking.
- Emailed digests.
- GitHub Pages / generated HTML site, if plain markdown ever feels insufficient.
- A second LLM provider (e.g. Anthropic), if Gemini alone proves limiting.
- Trained (non-heuristic) ranking model, once enough feedback accumulates to make one worthwhile.
