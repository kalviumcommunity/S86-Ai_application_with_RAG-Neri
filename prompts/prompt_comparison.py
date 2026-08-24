import os
import logging

from dotenv import load_dotenv
from openai import (
    OpenAI,
    AuthenticationError,
    RateLimitError,
    APIStatusError
)

from answer import SYSTEM_TEMPLATE, ANSWER_TEMPLATE_V1, render


# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# Configure logging
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# --------------------------------------------------
# Read configuration from .env
# --------------------------------------------------

base_url = os.getenv("OPENAI_BASE_URL")
api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("CHAT_MODEL")


# --------------------------------------------------
# Validate configuration
# --------------------------------------------------

if not base_url:
    raise ValueError("OPENAI_BASE_URL is missing from .env")

if not api_key:
    raise ValueError("OPENAI_API_KEY is missing from .env")

if not model:
    raise ValueError("CHAT_MODEL is missing from .env")


# --------------------------------------------------
# Create OpenAI-compatible client
# --------------------------------------------------

client = OpenAI(
    base_url=base_url,
    api_key=api_key
)


# --------------------------------------------------
# Task 3:
# Two prompt variations for the same task
# --------------------------------------------------

context_text = (
    "Policy context:\n"
    "- Refund requests are accepted within 30 days of purchase.\n"
    "- Proof of payment is required.\n"
    "- Shipping charges are non-refundable."
)

prompt_1 = render(
    ANSWER_TEMPLATE_V1,
    context=context_text,
    question="Explain our refund policy."
)

prompt_2 = render(
    ANSWER_TEMPLATE_V1,
    context=context_text,
    question=(
        "In one sentence, state the refund window in days."
    )
)


# --------------------------------------------------
# Function to call the model
# --------------------------------------------------

def get_response(user_prompt):
    messages = [
        {
            "role": "system",
            "content": SYSTEM_TEMPLATE
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    logging.info("REQUEST: %s", messages)

    response = client.chat.completions.create(
        model=model,
        messages=messages
    )

    answer = response.choices[0].message.content

    logging.info("RESPONSE: %s", answer)

    if response.usage:
        logging.info("USAGE: %s", response.usage)

    return answer


# --------------------------------------------------
# Main program
# --------------------------------------------------

try:

    print("=" * 60)
    print("PROMPT CONSTRUCTION AND COMPARISON")
    print("=" * 60)


    # --------------------------------------------------
    # Prompt 1 - Vague
    # --------------------------------------------------

    print("\nPROMPT 1 - VAGUE")
    print("-" * 60)
    print("Question: Explain our refund policy.")

    response_1 = get_response(prompt_1)

    print("\nRESPONSE 1")
    print("-" * 60)
    print(response_1)


    # --------------------------------------------------
    # Prompt 2 - Clear and constrained
    # --------------------------------------------------

    print("\nPROMPT 2 - CLEAR AND CONSTRAINED")
    print("-" * 60)
    print("Question: In one sentence, state the refund window in days.")

    response_2 = get_response(prompt_2)

    print("\nRESPONSE 2")
    print("-" * 60)
    print(response_2)


    # --------------------------------------------------
    # Task 4 - Prompt comparison
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("PROMPT COMPARISON")
    print("=" * 60)

    print("""
Prompt 1 is vague because it does not specify the expected
response format, length, or what should happen if the required
information is unavailable.

Prompt 2 is clearer because it specifies the exact information
required, limits the response to one sentence, and provides a
fallback when the information is unavailable.
""")

    print("CHOSEN PROMPT:")
    print(prompt_2)

    print("""
REASON:
Prompt 2 is more reliable because it clearly defines the task,
expected format, response length, and fallback behavior. This
reduces ambiguity and makes the model response more consistent.
""")


# --------------------------------------------------
# Error handling
# --------------------------------------------------

except AuthenticationError:
    print(
        "\nAuthentication failed (401): "
        "Check OPENAI_API_KEY in your .env file."
    )

except RateLimitError:
    print(
        "\nRate limit or quota reached (429): "
        "Please wait and retry."
    )

except APIStatusError as error:

    if error.status_code == 503:
        print(
            "\nService temporarily unavailable (503): "
            "The model is currently busy. Please try again later."
        )
    else:
        print(
            f"\nAPI error ({error.status_code}): {error}"
        )

except Exception as error:
    print(
        f"\nAn unexpected error occurred: {error}"
    )