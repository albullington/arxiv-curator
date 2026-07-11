import time

from google.genai import errors

RETRYABLE_CODES = {429, 500, 502, 503, 504}
DEFAULT_MAX_ATTEMPTS = 4
DEFAULT_BASE_DELAY = 2.0


def with_retries(
    func,
    *args,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    base_delay: float = DEFAULT_BASE_DELAY,
    **kwargs,
):
    attempt = 0
    while True:
        try:
            return func(*args, **kwargs)
        except errors.APIError as exc:
            attempt += 1
            if exc.code not in RETRYABLE_CODES or attempt >= max_attempts:
                raise
            time.sleep(base_delay * (2 ** (attempt - 1)))
