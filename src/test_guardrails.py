"""
Test cases for hallucination guardrails.

Demonstrates:
1. Strong retrieval -> answer
2. Weak retrieval -> refusal
3. Empty retrieval -> refusal
"""

from pprint import pprint

from src.grounded_generation import (
    retrieve_chunks,
    generate_grounded_answer,
)

from src.guardrails import (
    check_retrieval,
    get_strong_chunks,
    refusal_response,
    MIN_TOP_SCORE,
)


# ---------------------------------------------------------
# Test 1: Strong context
# ---------------------------------------------------------

def test_strong_context():

    print("\n" + "=" * 70)
    print("TEST 1: STRONG RETRIEVAL")
    print("=" * 70)

    question = (
        "What should a technician do if abnormal vibration "
        "is detected?"
    )

    chunks = retrieve_chunks(
        question,
        k=3,
    )

    print("\nRetrieved chunks:", len(chunks))

    for chunk in chunks:
        print(
            f"  {chunk['metadata'].get('source')} "
            f"| similarity={chunk.get('similarity', 0):.4f}"
        )

    check = check_retrieval(
        chunks,
        min_score=MIN_TOP_SCORE,
    )

    print("\nGuardrail result:")
    pprint(check)

    if check["allowed"]:

        strong_chunks = get_strong_chunks(
            chunks,
            min_score=MIN_TOP_SCORE,
        )

        answer = generate_grounded_answer(
            question,
            strong_chunks[:3],
        )

        print("\nSTATUS: ANSWERED")
        print("\nAnswer:")
        print(answer["answer"])

    else:

        print("\nSTATUS: REFUSED")
        print(
            refusal_response(
                check["reason"]
            )["answer"]
        )


# ---------------------------------------------------------
# Test 2: Unsupported question
# ---------------------------------------------------------

def test_unsupported_context():

    print("\n" + "=" * 70)
    print("TEST 2: UNSUPPORTED QUESTION")
    print("=" * 70)

    question = (
        "What is the company's refund policy for "
        "customers returning products?"
    )

    chunks = retrieve_chunks(
        question,
        k=3,
    )

    print("\nRetrieved chunks:", len(chunks))

    for chunk in chunks:
        print(
            f"  {chunk['metadata'].get('source')} "
            f"| similarity={chunk.get('similarity', 0):.4f}"
        )

    check = check_retrieval(
        chunks,
        min_score=MIN_TOP_SCORE,
    )

    print("\nGuardrail result:")
    pprint(check)

    if check["allowed"]:

        strong_chunks = get_strong_chunks(
            chunks,
            min_score=MIN_TOP_SCORE,
        )

        answer = generate_grounded_answer(
            question,
            strong_chunks[:3],
        )

        print("\nSTATUS: ANSWERED")
        print(answer["answer"])

    else:

        refusal = refusal_response(
            check["reason"]
        )

        print("\nSTATUS: REFUSED")
        print("\nAnswer:")
        print(refusal["answer"])


# ---------------------------------------------------------
# Test 3: Empty retrieval
# ---------------------------------------------------------

def test_empty_context():

    print("\n" + "=" * 70)
    print("TEST 3: EMPTY CONTEXT")
    print("=" * 70)

    chunks = []

    check = check_retrieval(
        chunks,
        min_score=MIN_TOP_SCORE,
    )

    print("\nGuardrail result:")
    pprint(check)

    refusal = refusal_response(
        check["reason"]
    )

    print("\nSTATUS: REFUSED")
    print("\nAnswer:")
    print(refusal["answer"])


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    print("\n")
    print("#" * 70)
    print("# HALLUCINATION GUARDRAIL EVALUATION")
    print("#" * 70)

    test_strong_context()

    test_unsupported_context()

    test_empty_context()

    print("\n")
    print("#" * 70)
    print("# ALL GUARDRAIL TESTS COMPLETED")
    print("#" * 70)