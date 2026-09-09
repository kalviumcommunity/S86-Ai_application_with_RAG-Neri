import json
import logging
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.grounded_generation import guarded_answer


logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG API",
    description="Backend API for the grounded RAG system",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=1000,
        description="Question to ask the RAG system",
    )


def make_event(event_type: str, **data) -> str:
    """
    Convert an event into Server-Sent Events format.
    """
    event = {
        "type": event_type,
        **data,
    }

    return f"data: {json.dumps(event)}\n\n"


@app.get("/")
def root():
    return {
        "service": "RAG API",
        "status": "running",
    }


@app.post("/query")
def query_rag(request: QueryRequest):
    """
    Normal non-streaming RAG endpoint.
    """
    try:
        result = guarded_answer(request.question)

        return {
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "status": result.get("status", "answered"),
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception as error:
        logger.exception("RAG query failed")

        raise HTTPException(
            status_code=500,
            detail="RAG service failed",
        )


@app.post("/query/stream")
async def stream_query(request: QueryRequest):
    """
    Streaming RAG endpoint.

    Events:
    - citations
    - token
    - done
    - error
    """

    async def event_generator() -> AsyncGenerator[str, None]:

        try:
            # --------------------------------------------------
            # Run the existing guarded RAG pipeline
            # --------------------------------------------------
            result = guarded_answer(
                request.question,
                candidate_k=10,
                final_k=3,
            )

            status = result.get("status", "answered")

            # --------------------------------------------------
            # Refusal / weak context
            # --------------------------------------------------
            if status != "answered":
                yield make_event(
                    "error",
                    message=result.get(
                        "answer",
                        "I don't have enough reliable context to answer that.",
                    ),
                )

                yield make_event(
                    "done",
                    status=status,
                )

                return

            # --------------------------------------------------
            # Prepare citations
            # --------------------------------------------------
            sources = []

            for index, chunk in enumerate(
                result.get("retrieved_chunks", []),
                start=1,
            ):

                metadata = chunk.get("metadata", {})

                sources.append(
                    {
                        "id": f"source-{index}",
                        "label": f"[{index}]",
                        "document": metadata.get(
                            "source",
                            "Unknown source",
                        ),
                        "chunk_id": chunk.get(
                            "id",
                            metadata.get(
                                "chunk_id",
                                f"chunk-{index}",
                            ),
                        ),
                        "chunk_index": metadata.get(
                            "chunk_index"
                        ),
                        "section": metadata.get(
                            "section"
                        ),
                        "score": chunk.get(
                            "similarity",
                            chunk.get("score"),
                        ),
                        "text": chunk.get("text", ""),
                    }
                )

            # --------------------------------------------------
            # Send citations before answer
            # --------------------------------------------------
            yield make_event(
                "citations",
                sources=sources,
            )

            # --------------------------------------------------
            # Get generated answer
            # --------------------------------------------------
            answer = result.get("answer", "")

            # --------------------------------------------------
            # Simulate progressive streaming
            #
            # IMPORTANT:
            # This streams the already-grounded answer
            # progressively. It does not generate new content
            # independently of the RAG pipeline.
            # --------------------------------------------------
            words = answer.split(" ")

            for index, word in enumerate(words):

                text = word

                if index < len(words) - 1:
                    text += " "

                yield make_event(
                    "token",
                    text=text,
                )

            # --------------------------------------------------
            # Finish
            # --------------------------------------------------
            yield make_event(
                "done",
                status="answered",
            )

        except ValueError as error:

            logger.warning(
                "Invalid streaming request: %s",
                error,
            )

            yield make_event(
                "error",
                message=str(error),
            )

        except Exception:

            logger.exception(
                "Streaming RAG request failed"
            )

            yield make_event(
                "error",
                message=(
                    "The answer stopped streaming. "
                    "Please try again."
                ),
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )