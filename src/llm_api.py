import os
import logging
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import (
    OpenAI,
    AuthenticationError,
    RateLimitError,
    APIStatusError
)


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from prompts.answer import SYSTEM_TEMPLATE, ANSWER_TEMPLATE_V2, render


# Load environment variables from .env
load_dotenv()


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# Read configuration from environment variables
base_url = os.getenv("OPENAI_BASE_URL")
api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("CHAT_MODEL")


# Validate required configuration
if not base_url:
    raise ValueError("OPENAI_BASE_URL is missing from .env")

if not api_key:
    raise ValueError("OPENAI_API_KEY is missing from .env")

if not model:
    raise ValueError("CHAT_MODEL is missing from .env")


# Create OpenAI-compatible client
client = OpenAI(
    base_url=base_url,
    api_key=api_key
)


# Messages sent to the language model
context_text = (
    "Policy excerpt: Customers can request a refund within 30 days of "
    "purchase with proof of payment."
)

question_text = "What is the refund window?"

messages = [
    {
        "role": "system",
        "content": SYSTEM_TEMPLATE
    },
    {
        "role": "user",
        "content": render(
            ANSWER_TEMPLATE_V2,
            context=context_text,
            question=question_text
        )
    }
]


def parse_json_response(raw, required=("answer", "source")):
    if raw is None:
        return None, "empty response"

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None, "malformed JSON"

    if not isinstance(data, dict):
        return None, "response is not a JSON object"

    missing = [key for key in required if key not in data]
    if missing:
        return None, f"missing fields: {missing}"

    invalid = [
        key for key in required
        if not isinstance(data[key], str) or not data[key].strip()
    ]
    if invalid:
        return None, f"invalid field values: {invalid}"

    return data, None


def request_structured_output(base_messages):
    last_error = "unknown error"
    last_raw = ""

    for attempt in range(2):
        request_messages = list(base_messages)

        if attempt == 1:
            request_messages.append({
                "role": "user",
                "content": (
                    "Your previous reply was invalid. "
                    "Return valid JSON only with keys answer and source."
                )
            })

        request_parameters = {
            "model": model,
            "messages": request_messages,
            "temperature": 0,
            "response_format": {"type": "json_object"}
        }

        logging.info(
            "REQUEST (attempt %s): %s",
            attempt + 1,
            request_messages
        )

        response = client.chat.completions.create(
            **request_parameters
        )

        raw = response.choices[0].message.content
        last_raw = raw or ""

        parsed, parse_error = parse_json_response(last_raw)
        if parsed:
            return parsed, response, last_raw, None

        last_error = parse_error
        logging.warning(
            "PARSE ERROR (attempt %s): %s | RAW: %s",
            attempt + 1,
            parse_error,
            last_raw
        )

    return None, None, last_raw, last_error


try:
    structured_data, response, raw, parse_error = request_structured_output(
        messages
    )

    if structured_data is None:
        print("\nNERI Structured Response:")
        print(f"recover: {parse_error}")
        print("Raw model output:")
        print(raw)
    else:
        # Log the validated JSON payload.
        logging.info("RESPONSE JSON: %s", structured_data)

        # Log token usage if available.
        if response and response.usage:
            logging.info("USAGE: %s", response.usage)

        print("\nNERI Structured Response:")
        print(f"Answer: {structured_data['answer']}")
        print(f"Source: {structured_data['source']}")


except AuthenticationError:
    # 401 Unauthorized
    print(
        "Authentication failed (401): "
        "Check OPENAI_API_KEY in your .env file."
    )


except RateLimitError:
    # 429 Too Many Requests
    print(
        "Rate limit or quota reached (429): "
        "Please wait and retry."
    )


except APIStatusError as error:
    # Handle other API status errors such as 503
    if error.status_code == 503:
        print(
            "Service temporarily unavailable (503): "
            "The model is currently busy. Please try again later."
        )
    else:
        print(
            f"API error ({error.status_code}): {error}"
        )


except Exception as error:
    # Handle unexpected errors
    print(
        f"An unexpected error occurred: {error}"
    )