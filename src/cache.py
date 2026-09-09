import hashlib
import time


CACHE_TTL_SECONDS = 15 * 60


_query_cache = {}


def cache_key(question, filters=None):
    """
    Create a stable cache key from the question and filters.
    """

    raw = {
        "question": question.strip().lower(),
        "filters": filters or {},
    }

    return hashlib.sha256(
        str(raw).encode("utf-8")
    ).hexdigest()


def get_cached_answer(question, filters=None):
    """
    Return cached response if available and not expired.
    """

    key = cache_key(
        question,
        filters,
    )

    cached = _query_cache.get(key)

    if not cached:
        return None

    # Check TTL
    if (
        time.time() - cached["created_at"]
        > CACHE_TTL_SECONDS
    ):
        _query_cache.pop(key, None)
        return None

    return cached["response"]


def save_cached_answer(
    question,
    response,
    filters=None,
):
    """
    Store a response in the cache.
    """

    key = cache_key(
        question,
        filters,
    )

    _query_cache[key] = {
        "created_at": time.time(),
        "response": response,
    }


def clear_cache():
    """
    Clear all cached responses.
    """

    _query_cache.clear()


def cache_size():
    """
    Return the number of cached entries.
    """

    return len(_query_cache)