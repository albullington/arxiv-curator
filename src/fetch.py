import re
from typing import Optional

import feedparser
import requests

from arxiv_curator import db
from arxiv_curator.models import Paper

ARXIV_API_URL = "http://export.arxiv.org/api/query"

PAGE_COUNT_PATTERN = re.compile(r"(\d+)\s*pages?\b", re.IGNORECASE)


def extract_page_count(comment: Optional[str]) -> Optional[int]:
    if not comment:
        return None
    match = PAGE_COUNT_PATTERN.search(comment)
    return int(match.group(1)) if match else None


def build_query_url(categories: list[str], max_results: int) -> str:
    cat_query = "+OR+".join(f"cat:{c}" for c in categories)
    return (
        f"{ARXIV_API_URL}?search_query={cat_query}"
        f"&sortBy=submittedDate&sortOrder=descending&max_results={max_results}"
    )


def normalize_arxiv_id(arxiv_id_or_url: str) -> str:
    if "arxiv.org" in arxiv_id_or_url:
        return arxiv_id_or_url.rstrip("/").split("/")[-1]
    return arxiv_id_or_url


def build_id_query_url(arxiv_id: str) -> str:
    return f"{ARXIV_API_URL}?id_list={arxiv_id}"


def build_ids_query_url(arxiv_ids: list[str]) -> str:
    return f"{ARXIV_API_URL}?id_list={','.join(arxiv_ids)}"


def fetch_paper_by_id(arxiv_id: str) -> Optional[Paper]:
    url = build_id_query_url(arxiv_id)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    papers = parse_feed(response.text)
    return papers[0] if papers else None


def parse_feed(raw_text: str) -> list[Paper]:
    feed = feedparser.parse(raw_text)
    papers = []
    for entry in feed.entries:
        arxiv_id = entry.id.split("/abs/")[-1]
        authors = ", ".join(a.name for a in entry.get("authors", []))
        categories = ", ".join(t.term for t in entry.get("tags", []))
        papers.append(Paper(
            arxiv_id=arxiv_id,
            title=entry.title.replace("\n", " ").strip(),
            authors=authors,
            abstract=entry.summary.replace("\n", " ").strip(),
            categories=categories,
            published=entry.published,
            url=entry.link,
            pages=extract_page_count(entry.get("arxiv_comment")),
        ))
    return papers


def fetch_papers(categories: list[str], max_results: int) -> list[Paper]:
    url = build_query_url(categories, max_results)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return parse_feed(response.text)


def fetch_and_store(conn, categories: list[str], max_results: int) -> int:
    papers = fetch_papers(categories, max_results)
    new_count = 0
    for paper in papers:
        if not db.paper_exists(conn, paper.arxiv_id):
            db.insert_paper(conn, paper, source="fetch")
            new_count += 1
    return new_count


BACKFILL_CHUNK_SIZE = 50


def backfill_pages(conn) -> int:
    missing = db.papers_missing_pages(conn)
    updated_count = 0
    for i in range(0, len(missing), BACKFILL_CHUNK_SIZE):
        chunk = missing[i:i + BACKFILL_CHUNK_SIZE]
        url = build_ids_query_url([p.arxiv_id for p in chunk])
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        for paper in parse_feed(response.text):
            if paper.pages is not None:
                db.update_paper_pages(conn, paper.arxiv_id, paper.pages)
                updated_count += 1
    return updated_count
