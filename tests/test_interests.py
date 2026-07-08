import pytest
from arxiv_curator.interests import load_interest_profile, profile_to_text


def test_load_full_profile(tmp_path):
    path = tmp_path / "interests.yaml"
    path.write_text(
        "summary: I like retrieval-augmented generation and agent evaluation.\n"
        "topics:\n  - RAG\n  - agents\n"
        "keywords:\n  - retrieval\n  - evaluation\n"
        "liked_examples:\n  - \"Self-RAG\"\n"
    )
    profile = load_interest_profile(path)
    assert "retrieval-augmented" in profile.summary
    assert profile.topics == ["RAG", "agents"]
    assert profile.keywords == ["retrieval", "evaluation"]
    assert profile.liked_examples == ["Self-RAG"]


def test_load_minimal_profile_defaults_lists(tmp_path):
    path = tmp_path / "interests.yaml"
    path.write_text("summary: Just a summary.\n")
    profile = load_interest_profile(path)
    assert profile.topics == []
    assert profile.keywords == []
    assert profile.liked_examples == []


def test_missing_summary_raises(tmp_path):
    path = tmp_path / "interests.yaml"
    path.write_text("topics:\n  - RAG\n")
    with pytest.raises(ValueError, match="summary"):
        load_interest_profile(path)


def test_profile_to_text_includes_all_sections():
    from arxiv_curator.interests import InterestProfile
    profile = InterestProfile(
        summary="I like RAG.", topics=["RAG"], keywords=["retrieval"], liked_examples=["Self-RAG"],
    )
    text = profile_to_text(profile)
    assert "I like RAG." in text
    assert "RAG" in text
    assert "retrieval" in text
    assert "Self-RAG" in text
