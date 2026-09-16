import re


# ============================================================
# Safety-Critical Terms
# ============================================================

SAFETY_PHRASES = {
    "electrical spark",
    "electric spark",
    "electrical sparking",
    "electric sparking",
    "electric shock",
    "electrical shock",
    "exposed wiring",
    "exposed wire",
    "exposed electrical",
    "smoke",
    "fire",
    "gas leak",
    "chemical leak",
    "chemical exposure",
    "serious injury",
    "injury",
    "emergency",
    "unsafe",
    "danger",
    "hazard",
    "entrapment",
    "uncontrolled movement",
}


SAFETY_WORDS = {
    "spark",
    "sparking",
    "smoke",
    "fire",
    "shock",
    "injury",
    "emergency",
    "unsafe",
    "danger",
    "hazard",
    "entrapment",
}


# ============================================================
# Safety Detection
# ============================================================

def _contains_phrase(
    text: str,
    phrase: str,
) -> bool:
    """
    Check whether a complete safety phrase exists
    in the text.
    """

    pattern = (
        r"\b"
        + re.escape(phrase)
        + r"\b"
    )

    return re.search(
        pattern,
        text,
        flags=re.IGNORECASE,
    ) is not None


def is_safety_critical(
    query: str,
) -> bool:
    """
    Determine whether a troubleshooting request
    contains a safety-critical condition.

    Normal operational problems such as:
        - low pressure
        - pressure fluctuation
        - overheating
        - slow operation

    are NOT automatically treated as safety-critical.

    Explicit hazards such as:
        - electrical spark
        - smoke
        - fire
        - electric shock
        - exposed wiring
        - injury
        - emergency

    are treated as safety-critical.
    """

    if not query or not query.strip():
        return False

    query_lower = query.lower()

    # Check multi-word safety phrases first.
    for phrase in SAFETY_PHRASES:

        if _contains_phrase(
            query_lower,
            phrase,
        ):
            return True

    # Check individual safety words.
    for word in SAFETY_WORDS:

        if _contains_phrase(
            query_lower,
            word,
        ):
            return True

    # Hydraulic leaks can create a safety hazard.
    if (
        _contains_phrase(
            query_lower,
            "hydraulic leak",
        )
        or _contains_phrase(
            query_lower,
            "pressurized leak",
        )
    ):
        return True

    return False


# ============================================================
# Evidence Checks
# ============================================================

def has_safety_evidence(
    retrieved_chunks: list[dict],
) -> bool:
    """
    Check whether retrieved evidence contains
    an approved safety document.
    """

    for chunk in retrieved_chunks:

        metadata = chunk.get(
            "metadata",
            {},
        )

        document_type = metadata.get(
            "document_type",
            "",
        )

        if document_type == "safety":
            return True

    return False


def has_reliable_evidence(
    retrieved_chunks: list[dict],
) -> bool:
    """
    Check whether usable documentation
    was retrieved.
    """

    if not retrieved_chunks:
        return False

    for chunk in retrieved_chunks:

        text = chunk.get(
            "text",
            "",
        ).strip()

        if text:
            return True

    return False


# ============================================================
# No-Answer Responses
# ============================================================

def build_no_answer_response() -> dict:
    """
    Build the standard response when Neri
    cannot find enough approved documentation.
    """

    return {
        "reliable": False,
        "message": (
            "We couldn't find enough information in "
            "approved documentation to provide a reliable answer."
        ),
        "possible_causes": [],
        "safety_warning": None,
        "troubleshooting_steps": [],
        "sources": [],
    }


def build_safety_evidence_response() -> dict:
    """
    Build a response when a request is safety-critical
    but approved safety documentation is unavailable.
    """

    return {
        "reliable": False,
        "message": (
            "This request may involve a safety-critical "
            "condition, but no approved safety documentation "
            "was found. Neri cannot provide safety instructions "
            "without supporting safety documentation."
        ),
        "possible_causes": [],
        "safety_warning": (
            "Follow the organization's approved emergency "
            "and safety procedures before continuing."
        ),
        "troubleshooting_steps": [],
        "sources": [],
    }