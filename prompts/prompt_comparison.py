import os
import logging

from dotenv import load_dotenv
from openai import (
    OpenAI,
    AuthenticationError,
    RateLimitError,
    APIStatusError
)


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
# Task 2:
# System message with role, scope, constraints,
# and fallback
# --------------------------------------------------

system_prompt = """
You are a concise and factual assistant.

Your role is to answer staff questions clearly and accurately.

Only provide information that you can support from the information
available in the conversation.

Do not invent facts or make unsupported claims.

Keep your responses concise and professional.

If you do not have enough information to answer reliably, say:
"I don't know based on the available information."
"""


# --------------------------------------------------
# Task 3:
# Two prompt variations for the same task
# --------------------------------------------------

prompt_1 = "Explain our refund policy."

prompt_2 = (
    "In one sentence, state the refund window in days. "
    "If the refund window is not provided, say that you don't know."
)


# --------------------------------------------------
# Function to call the model
# --------------------------------------------------

def get_response(user_prompt):
    messages = [
        {
            "role": "system",
            "content": system_prompt
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
    print(prompt_1)

    response_1 = get_response(prompt_1)

    print("\nRESPONSE 1")
    print("-" * 60)
    print(response_1)


    # --------------------------------------------------
    # Prompt 2 - Clear and constrained
    # --------------------------------------------------

    print("\nPROMPT 2 - CLEAR AND CONSTRAINED")
    print("-" * 60)
    print(prompt_2)

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