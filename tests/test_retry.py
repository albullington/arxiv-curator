import pytest
from google.genai import errors

from arxiv_curator.llm.retry import with_retries


def test_with_retries_returns_result_on_first_success():
    def succeeds():
        return "ok"

    assert with_retries(succeeds) == "ok"


def test_with_retries_retries_on_retryable_error_then_succeeds(monkeypatch):
    monkeypatch.setattr("arxiv_curator.llm.retry.time.sleep", lambda seconds: None)
    calls = {"count": 0}

    def flaky():
        calls["count"] += 1
        if calls["count"] < 3:
            raise errors.APIError(503, {"error": {"message": "high demand"}})
        return "recovered"

    result = with_retries(flaky, max_attempts=4, base_delay=0.01)
    assert result == "recovered"
    assert calls["count"] == 3


def test_with_retries_raises_after_max_attempts(monkeypatch):
    monkeypatch.setattr("arxiv_curator.llm.retry.time.sleep", lambda seconds: None)

    def always_fails():
        raise errors.APIError(429, {"error": {"message": "rate limited"}})

    with pytest.raises(errors.APIError):
        with_retries(always_fails, max_attempts=3, base_delay=0.01)


def test_with_retries_does_not_retry_non_retryable_code(monkeypatch):
    calls = {"count": 0}

    def bad_request():
        calls["count"] += 1
        raise errors.APIError(400, {"error": {"message": "bad request"}})

    with pytest.raises(errors.APIError):
        with_retries(bad_request, max_attempts=4, base_delay=0.01)
    assert calls["count"] == 1
