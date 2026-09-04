"""OpenAI-compatible LLM API utilities for grounded RAG generation."""

import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import (
    OpenAI,
    AuthenticationError,
    RateLimitError,
    APIStatusError,
)

# -------------------------------------------------------------------
# Project root
# -------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# -------------------------------------------------------------------
# Prompts
# -------------------------------------------------------------------

from prompts.answer import (
    SYSTEM_TEMPLATE,
    ANSWER_TEMPLATE_V2,
    render,
)


# -------------------------------------------------------------------
# Environment
# -------------------------------------------------------------------

load_dotenv(ROOT_DIR / ".env")


# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

BASE_URL = os.getenv("OPENAI_BASE_URL")
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("CHAT_MODEL")


def _validate_config():
    """Validate required environment variables."""

    if not BASE_URL:
        raise ValueError(
            "OPENAI_BASE_URL is missing from .env"
        )

    if not API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is missing from .env"
        )

    if not MODEL:
        raise ValueError(
            "CHAT_MODEL is missing from .env"
        )


# -------------------------------------------------------------------
# OpenAI-compatible client
# -------------------------------------------------------------------

def get_client():
    """Create and return an OpenAI-compatible client."""

    _validate_config()

    return OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
    )


# -------------------------------------------------------------------
# JSON parsing
# -------------------------------------------------------------------

def parse_json_response(
    raw,
    required=("answer", "source"),
):
    """
    Parse and validate a JSON response.

    Returns:
        (data, error)
    """

    if raw is None:
        return None, "empty response"

    raw = raw.strip()

    if not raw:
        return None, "empty response"

    # First attempt: normal JSON
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:

        # Sometimes models return JSON inside markdown fences.
        cleaned = raw

        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]

        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]

        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            return None, "malformed JSON"

    if not isinstance(data, dict):
        return None, "response is not a JSON object"

    missing = [
        key
        for key in required
        if key not in data
    ]

    if missing:
        return None, f"missing fields: {missing}"

    invalid = [
        key
        for key in required
        if not isinstance(data[key], str)
        or not data[key].strip()
    ]

    if invalid:
        return None, f"invalid field values: {invalid}"

    return data, None


# -------------------------------------------------------------------
# Structured LLM request
# -------------------------------------------------------------------

def request_structured_output(
    base_messages,
    client=None,
    model=None,
    max_attempts=2,
):
    """
    Send messages to the LLM and return validated structured JSON.

    Returns:
        parsed_data, response, raw_response, error
    """

    if client is None:
        client = get_client()

    model = model or MODEL

    last_error = "unknown error"
    last_raw = ""

    for attempt in range(max_attempts):

        request_messages = list(base_messages)

        # Retry instruction if first response was invalid.
        if attempt > 0:
            request_messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous reply was invalid. "
                        "Return valid JSON only with exactly "
                        "these keys: answer and source."
                    ),
                }
            )

        request_parameters = {
            "model": model,
            "messages": request_messages,
            "temperature": 0,
            "response_format": {
                "type": "json_object"
            },
        }

        logging.info(
            "LLM REQUEST (attempt %s)",
            attempt + 1,
        )

        try:

            response = client.chat.completions.create(
                **request_parameters
            )

            if not response.choices:
                last_error = "LLM returned no choices"
                continue

            raw = response.choices[0].message.content

            last_raw = raw or ""

            parsed, parse_error = parse_json_response(
                last_raw
            )

            if parsed is not None:

                logging.info(
                    "LLM RESPONSE JSON: %s",
                    parsed,
                )

                if response.usage:
                    logging.info(
                        "LLM USAGE: %s",
                        response.usage,
                    )

                return (
                    parsed,
                    response,
                    last_raw,
                    None,
                )

            last_error = parse_error

            logging.warning(
                "JSON PARSE ERROR (attempt %s): %s",
                attempt + 1,
                parse_error,
            )

        except AuthenticationError:

            raise

        except RateLimitError:

            raise

        except APIStatusError:

            raise

        except Exception as error:

            last_error = str(error)

            logging.exception(
                "Unexpected LLM error"
            )

    return (
        None,
        None,
        last_raw,
        last_error,
    )


# -------------------------------------------------------------------
# Main RAG generation function
# -------------------------------------------------------------------

def generate_answer(
    query: str,
    context: str,
    client=None,
    model=None,
) -> str:
    """
    Generate a grounded answer using the supplied context.

    Args:
        query:
            User's question.

        context:
            Retrieved RAG context.

        client:
            Optional OpenAI-compatible client.

        model:
            Optional model override.

    Returns:
        Answer string.
    """

    if not query or not query.strip():
        raise ValueError(
            "query must not be empty"
        )

    if not context or not context.strip():
        return "I do not know based on the provided context."

    messages = [
        {
            "role": "system",
            "content": SYSTEM_TEMPLATE,
        },
        {
            "role": "user",
            "content": render(
                ANSWER_TEMPLATE_V2,
                context=context,
                question=query,
            ),
        },
    ]

    try:

        structured_data, response, raw, error = (
            request_structured_output(
                messages,
                client=client,
                model=model,
            )
        )

        if structured_data is None:

            logging.warning(
                "Could not parse structured response: %s",
                error,
            )

            return (
                "I do not know based on the provided context."
            )

        answer = structured_data.get(
            "answer",
            "",
        ).strip()

        source = structured_data.get(
            "source",
            "",
        ).strip()

        if not answer:
            return (
                "I do not know based on the provided context."
            )

        # Return the answer in a format that the RAG
        # pipeline can consume.
        if source:
            return f"{answer} [Source: {source}]"

        return answer

    except AuthenticationError:

        return (
            "Authentication failed. "
            "Please check the API configuration."
        )

    except RateLimitError:

        return (
            "The LLM API rate limit has been reached. "
            "Please try again later."
        )

    except APIStatusError as error:

        if error.status_code == 503:
            return (
                "The LLM service is temporarily unavailable. "
                "Please try again later."
            )

        return (
            f"LLM API error ({error.status_code})."
        )

    except Exception as error:

        logging.exception(
            "Unexpected error while generating answer"
        )

        return (
            "I was unable to generate an answer."
        )


# -------------------------------------------------------------------
# Optional structured version
# -------------------------------------------------------------------

def generate_structured_answer(
    query: str,
    context: str,
    client=None,
    model=None,
) -> dict:
    """
    Generate and return the complete structured response.

    Returns:
        {
            "answer": "...",
            "source": "..."
        }
    """

    if not query or not query.strip():
        raise ValueError(
            "query must not be empty"
        )

    if not context or not context.strip():

        return {
            "answer": (
                "I do not know based on the provided context."
            ),
            "source": "",
        }

    messages = [
        {
            "role": "system",
            "content": SYSTEM_TEMPLATE,
        },
        {
            "role": "user",
            "content": render(
                ANSWER_TEMPLATE_V2,
                context=context,
                question=query,
            ),
        },
    ]

    try:

        structured_data, response, raw, error = (
            request_structured_output(
                messages,
                client=client,
                model=model,
            )
        )

        if structured_data is None:

            return {
                "answer": (
                    "I do not know based on the provided context."
                ),
                "source": "",
            }

        return structured_data

    except AuthenticationError:

        return {
            "answer": (
                "Authentication failed. "
                "Please check the API configuration."
            ),
            "source": "",
        }

    except RateLimitError:

        return {
            "answer": (
                "The LLM API rate limit has been reached."
            ),
            "source": "",
        }

    except APIStatusError as error:

        return {
            "answer": (
                f"LLM API error ({error.status_code})."
            ),
            "source": "",
        }

    except Exception:

        logging.exception(
            "Unexpected structured generation error"
        )

        return {
            "answer": (
                "I was unable to generate an answer."
            ),
            "source": "",
        }


# -------------------------------------------------------------------
# Standalone test
# -------------------------------------------------------------------

def main():
    """Run a simple standalone LLM API test."""

    context = (
        "Policy excerpt: Customers can request a refund "
        "within 30 days of purchase with proof of payment."
    )

    question = "What is the refund window?"

    print("=" * 70)
    print("LLM API TEST")
    print("=" * 70)

    result = generate_structured_answer(
        question,
        context,
    )

    print("\nNERI Structured Response:")
    print(
        json.dumps(
            result,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()