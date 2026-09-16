import hashlib
import time


_cache: dict[str, tuple[float, dict]] = {}


def _create_key(
    machine: str,
    machine_id: str,
    problem: str,
    error_code: str | None,
) -> str:
    """Create a unique cache key for a troubleshooting request."""

    raw_key = "|".join(
        [
            machine.strip().lower(),
            machine_id.strip().lower(),
            problem.strip().lower(),
            (error_code or "").strip().lower(),
        ]
    )

    return hashlib.sha256(
        raw_key.encode("utf-8")
    ).hexdigest()


def get_cached_response(
    machine: str,
    machine_id: str,
    problem: str,
    error_code: str | None,
    ttl_seconds: int,
) -> dict | None:
    """Return a cached response if it is still valid."""

    key = _create_key(
        machine,
        machine_id,
        problem,
        error_code,
    )

    cached = _cache.get(key)

    if cached is None:
        return None

    created_at, response = cached

    if time.time() - created_at > ttl_seconds:
        del _cache[key]
        return None

    return response


def cache_response(
    machine: str,
    machine_id: str,
    problem: str,
    error_code: str | None,
    response: dict,
) -> None:
    """Store a troubleshooting response in the cache."""

    key = _create_key(
        machine,
        machine_id,
        problem,
        error_code,
    )

    _cache[key] = (
        time.time(),
        response,
    )