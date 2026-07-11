from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class InterestProfile:
    summary: str
    topics: list[str]
    keywords: list[str]
    liked_examples: list[str]


def load_interest_profile(path: Path) -> InterestProfile:
    data = yaml.safe_load(Path(path).read_text()) or {}
    if "summary" not in data or not data["summary"]:
        raise ValueError("interests.yaml must include a non-empty 'summary' field")
    return InterestProfile(
        summary=data["summary"],
        topics=data.get("topics") or [],
        keywords=data.get("keywords") or [],
        liked_examples=data.get("liked_examples") or [],
    )


def profile_to_text(profile: InterestProfile) -> str:
    parts = [profile.summary]
    if profile.topics:
        parts.append("Topics: " + ", ".join(profile.topics))
    if profile.keywords:
        parts.append("Keywords: " + ", ".join(profile.keywords))
    if profile.liked_examples:
        parts.append("Examples of papers I like: " + "; ".join(profile.liked_examples))
    return "\n".join(parts)
