import os
import logging

from dotenv import load_dotenv
from openai import (
    OpenAI,
    AuthenticationError,
    RateLimitError,
    APIStatusError
)


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
messages = [
    {
        "role": "system",
        "content": "You are a concise assistant."
    },
    {
        "role": "user",
        "content": "Explain what machine maintenance means in one sentence."
    }
]


try:
    # Log the outgoing request
    logging.info("REQUEST: %s", messages)

    # Send chat completion request
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )

    # Extract generated response
    answer = response.choices[0].message.content

    # Log the response
    logging.info("RESPONSE: %s", answer)

    # Log token usage if available
    if response.usage:
        logging.info("USAGE: %s", response.usage)

    # Print the generated answer
    print("\nNERI Response:")
    print(answer)


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