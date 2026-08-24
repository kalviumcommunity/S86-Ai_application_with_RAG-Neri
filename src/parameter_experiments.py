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
# Logging
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

base_url = os.getenv("OPENAI_BASE_URL")
api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("CHAT_MODEL")


if not base_url:
    raise ValueError("OPENAI_BASE_URL is missing from .env")

if not api_key:
    raise ValueError("OPENAI_API_KEY is missing from .env")

if not model:
    raise ValueError("CHAT_MODEL is missing from .env")


# --------------------------------------------------
# OpenAI-compatible client
# --------------------------------------------------

client = OpenAI(
    base_url=base_url,
    api_key=api_key
)


# --------------------------------------------------
# Common message
# --------------------------------------------------

messages = [
    {
        "role": "system",
        "content": (
            "You are a concise assistant. "
            "Answer clearly and accurately."
        )
    },
    {
        "role": "user",
        "content": (
            "Explain why regular maintenance is important "
            "for manufacturing equipment."
        )
    }
]


# --------------------------------------------------
# Helper function
# --------------------------------------------------

def make_request(
    label,
    temperature=None,
    max_tokens=None,
    stop=None,
    top_p=None
):
    print("\n" + "=" * 70)
    print(label)
    print("=" * 70)

    parameters = {
        "model": model,
        "messages": messages
    }

    if temperature is not None:
        parameters["temperature"] = temperature

    if max_tokens is not None:
        parameters["max_tokens"] = max_tokens

    if stop is not None:
        parameters["stop"] = stop

    if top_p is not None:
        parameters["top_p"] = top_p

    logging.info("REQUEST PARAMETERS: %s", parameters)

    try:

        response = client.chat.completions.create(
            **parameters
        )

        answer = response.choices[0].message.content

        print("Response:")
        print(answer)

        logging.info(
            "%s RESPONSE: %s",
            label,
            answer
        )

        if response.usage:
            logging.info(
                "%s USAGE: %s",
                label,
                response.usage
            )

        return answer

    except AuthenticationError:

        print(
            "Authentication failed (401): "
            "Check OPENAI_API_KEY in your .env file."
        )

    except RateLimitError:

        print(
            "Rate limit or quota reached (429): "
            "Please wait and retry."
        )

    except APIStatusError as error:

        print(
            f"API error ({error.status_code}): {error}"
        )

    except Exception as error:

        print(
            f"Unexpected error: {error}"
        )


# ==================================================
# TASK 1 — TEMPERATURE
# ==================================================

print("\n")
print("#" * 70)
print("TASK 1 - TEMPERATURE EXPERIMENT")
print("#" * 70)

print("\nSame prompt:")
print(messages[1]["content"])


make_request(
    "TEMPERATURE = 0.0",
    temperature=0.0
)


make_request(
    "TEMPERATURE = 1.0",
    temperature=1.0
)


# ==================================================
# TASK 2 — MAX TOKENS
# ==================================================

print("\n")
print("#" * 70)
print("TASK 2 - MAX TOKENS EXPERIMENT")
print("#" * 70)


long_messages = [
    {
        "role": "system",
        "content": (
            "You are a detailed assistant. "
            "Provide a comprehensive explanation."
        )
    },
    {
        "role": "user",
        "content": (
            "Explain manufacturing equipment maintenance in detail. "
            "Discuss preventive maintenance, inspections, lubrication, "
            "condition monitoring, maintenance records, safety procedures, "
            "common causes of equipment failure, and the importance of "
            "reducing unexpected downtime."
        )
    }
]


messages_backup = messages

messages = long_messages


make_request(
    "MAX TOKENS = 50",
    temperature=0.1,
    max_tokens=50
)


messages = messages_backup


# ==================================================
# TASK 3 — STOP PARAMETER
# ==================================================

print("\n")
print("#" * 70)
print("TASK 3 - STOP PARAMETER EXPERIMENT")
print("#" * 70)


stop_messages = [
    {
        "role": "system",
        "content": (
            "You are a concise maintenance assistant."
        )
    },
    {
        "role": "user",
        "content": (
            "Write three short maintenance tips. "
            "After the third tip, write END."
        )
    }
]


messages_backup = messages

messages = stop_messages


make_request(
    "STOP SEQUENCE = END",
    temperature=0.1,
    max_tokens=100,
    stop=["END"]
)


messages = messages_backup


# ==================================================
# TASK 4 — RECOMMENDED SETTINGS
# ==================================================

print("\n")
print("#" * 70)
print("TASK 4 - RECOMMENDED SETTINGS")
print("#" * 70)

print(
    """
Recommended settings for a grounded, factual task:

temperature = 0.1

A low temperature keeps responses focused, consistent,
and less likely to introduce unnecessary variation.

max_tokens = 300

A reasonable output limit prevents unnecessarily long
responses and helps control output token usage and cost.

stop = optional

A stop sequence can be useful when the response should
end at a specific boundary.

top_p = default

Temperature and top_p are alternative sampling controls,
so they generally should not both be tuned at the same time.
"""
)