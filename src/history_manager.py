import os
import logging

from dotenv import load_dotenv
from openai import (
    OpenAI,
    AuthenticationError,
    RateLimitError,
    APIStatusError
)

from token_cost_estimator import count_tokens


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
# Read configuration
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
# Token budget for demonstration
#
# A small budget is intentionally used so that
# trimming can be demonstrated with a short
# conversation.
# --------------------------------------------------

HISTORY_TOKEN_BUDGET = 300


# --------------------------------------------------
# System message
# --------------------------------------------------

SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "You are a concise assistant. "
        "Answer clearly and keep responses brief."
    )
}


# --------------------------------------------------
# Conversation history
#
# The system message must always remain.
# --------------------------------------------------

history = [SYSTEM_MESSAGE]


# --------------------------------------------------
# Count total tokens in message history
# --------------------------------------------------

def total_tokens(messages):
    return sum(
        count_tokens(message["content"])
        for message in messages
    )


# --------------------------------------------------
# Trim oldest complete conversation turns
#
# Always preserve the system message.
# --------------------------------------------------

def trim_history(messages, budget):
    removed_turns = 0

    while total_tokens(messages) > budget and len(messages) > 3:

        # Remove the oldest user + assistant pair.
        del messages[1:3]

        removed_turns += 1

    return removed_turns


# --------------------------------------------------
# Send one conversation turn
# --------------------------------------------------

def ask(user_message, turn_number):

    # Add the user's message to history.
    history.append({
        "role": "user",
        "content": user_message
    })

    tokens_before_trim = total_tokens(history)

    print(f"\nTURN {turn_number}")
    print("-" * 60)

    print(f"Tokens before trimming: {tokens_before_trim}")
    logging.info(
        "Turn %s - tokens before trimming: %s",
        turn_number,
        tokens_before_trim
    )

    # Trim history if it exceeds the budget.
    removed_turns = trim_history(
        history,
        HISTORY_TOKEN_BUDGET
    )

    tokens_after_trim = total_tokens(history)

    if removed_turns > 0:

        print(
            f"Trimmed {removed_turns} old conversation turn(s)."
        )

        print(
            f"Tokens after trimming: {tokens_after_trim}"
        )

        logging.info(
            "Turn %s - trimmed %s old turn(s)",
            turn_number,
            removed_turns
        )

        logging.info(
            "Turn %s - tokens after trimming: %s",
            turn_number,
            tokens_after_trim
        )

    else:

        print("No trimming required.")

        print(
            f"Tokens after trimming: {tokens_after_trim}"
        )

    # Make sure the history is within the budget.
    if tokens_after_trim > HISTORY_TOKEN_BUDGET:
        print(
            "Warning: history is still above the configured budget."
        )

    try:

        logging.info(
            "Turn %s - sending request with %s tokens",
            turn_number,
            tokens_after_trim
        )

        response = client.chat.completions.create(
            model=model,
            messages=history
        )

        answer = response.choices[0].message.content

        # Add assistant response to history.
        history.append({
            "role": "assistant",
            "content": answer
        })

        print("Assistant:")
        print(answer)

        logging.info(
            "Turn %s - response: %s",
            turn_number,
            answer
        )

        if response.usage:
            logging.info(
                "Turn %s - API usage: %s",
                turn_number,
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

        if error.status_code == 503:

            print(
                "Service temporarily unavailable (503): "
                "The model is currently busy."
            )

        else:

            print(
                f"API error ({error.status_code}): {error}"
            )

    except Exception as error:

        print(
            f"An unexpected error occurred: {error}"
        )


# --------------------------------------------------
# Demonstration conversation
#
# The messages are intentionally long so that the
# small 300-token budget is exceeded.
# --------------------------------------------------

conversation = [
    (
        "Explain why regular inspection of manufacturing equipment "
        "is important and describe how inspection can help identify "
        "potential problems before they become serious failures."
    ),

    (
        "Explain the importance of preventive maintenance schedules "
        "for industrial equipment and describe how regular servicing "
        "can reduce unexpected breakdowns and improve equipment "
        "reliability over time."
    ),

    (
        "Describe what can happen when maintenance activities are "
        "delayed for a long period and explain how neglected equipment "
        "can affect reliability, operating efficiency, and downtime."
    ),

    (
        "Explain why maintenance teams should keep accurate records "
        "of inspections, repairs, and servicing activities and how "
        "these records can help identify recurring equipment problems."
    ),

    (
        "Describe the importance of following manufacturer maintenance "
        "instructions and explain why technicians should avoid using "
        "unsupported procedures when servicing industrial equipment."
    ),

    (
        "Explain how monitoring equipment condition over time can help "
        "maintenance teams identify developing problems and decide "
        "when inspection or servicing may be required."
    ),

    (
        "Explain why maintenance personnel should consider equipment "
        "safety procedures before performing inspection or repair work "
        "and why documented safety instructions are important."
    )
]


# --------------------------------------------------
# Run the multi-turn conversation
# --------------------------------------------------

print("=" * 60)
print("NERI - MESSAGE HISTORY MANAGEMENT")
print("=" * 60)

print(f"Token budget: {HISTORY_TOKEN_BUDGET}")
print("Strategy: trim oldest complete user/assistant turns")
print("System message is always preserved.")

for number, user_message in enumerate(
    conversation,
    start=1
):

    ask(
        user_message,
        number
    )


# --------------------------------------------------
# Final history information
# --------------------------------------------------

print("\n" + "=" * 60)
print("FINAL HISTORY")
print("=" * 60)

print(
    f"Final history token count: "
    f"{total_tokens(history)}"
)

print(
    f"Final message count: "
    f"{len(history)}"
)

print("\nRoles currently in history:")

for message in history:

    print(
        f"- {message['role']}"
    )