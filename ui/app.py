import os

import requests
import streamlit as st
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

API_URL = os.getenv(
    "RAG_API_URL",
    "http://127.0.0.1:8000",
)

QUERY_ENDPOINT = f"{API_URL}/query"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="🤖",
    layout="centered",
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            color: #666;
            margin-bottom: 1.5rem;
        }

        .source-box {
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: 8px;
            margin-bottom: 8px;
        }

        .status-box {
            padding: 10px;
            border-radius: 8px;
            background-color: #f5f5f5;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🤖 RAG Knowledge Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Ask questions and get answers grounded in the knowledge base.'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# API STATUS
# ============================================================

with st.sidebar:

    st.header("⚙️ Configuration")

    st.write("Backend API:")

    st.code(API_URL)

    if st.button("Check API"):

        try:

            response = requests.get(
                f"{API_URL}/health",
                timeout=5,
            )

            if response.ok:

                st.success("Backend is online.")

            else:

                st.error(
                    f"Backend returned HTTP {response.status_code}"
                )

        except requests.RequestException:

            st.error(
                "Could not connect to the backend."
            )


# ============================================================
# QUESTION INPUT
# ============================================================

question = st.text_area(
    "Ask a question",
    placeholder=(
        "Example: What should a technician do "
        "if abnormal vibration is detected?"
    ),
    height=120,
)


# ============================================================
# SUBMIT BUTTON
# ============================================================

ask_button = st.button(
    "🔍 Ask Question",
    type="primary",
    use_container_width=True,
)


# ============================================================
# QUERY FUNCTION
# ============================================================

def ask_rag_api(question: str):

    response = requests.post(
        QUERY_ENDPOINT,
        json={
            "question": question,
        },
        timeout=120,
    )

    if not response.ok:

        try:
            error_detail = response.json().get(
                "detail",
                "RAG API request failed.",
            )
        except ValueError:
            error_detail = "RAG API request failed."

        raise RuntimeError(
            f"HTTP {response.status_code}: {error_detail}"
        )

    return response.json()


# ============================================================
# PROCESS QUESTION
# ============================================================

if ask_button:

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not question.strip():

        st.warning(
            "Please enter a question before submitting."
        )

    elif len(question.strip()) < 3:

        st.warning(
            "Question must contain at least 3 characters."
        )

    else:

        # ----------------------------------------------------
        # Call backend
        # ----------------------------------------------------

        try:

            with st.spinner(
                "Searching the knowledge base and generating an answer..."
            ):

                result = ask_rag_api(
                    question.strip()
                )

            # Save result
            st.session_state["last_result"] = result

        except requests.ConnectionError:

            st.error(
                "Could not connect to the RAG backend. "
                "Make sure the FastAPI server is running."
            )

        except requests.Timeout:

            st.error(
                "The request timed out. "
                "Please try again."
            )

        except RuntimeError as error:

            st.error(str(error))

        except Exception as error:

            st.error(
                f"Unexpected error: {error}"
            )


# ============================================================
# DISPLAY RESULT
# ============================================================

if "last_result" in st.session_state:

    result = st.session_state["last_result"]

    st.divider()

    # --------------------------------------------------------
    # Answer
    # --------------------------------------------------------

    st.subheader("💬 Answer")

    answer = result.get(
        "answer",
        "No answer returned.",
    )

    st.write(answer)

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    status = result.get(
        "status",
        "unknown",
    )

    if status == "answered":

        st.success(
            "Answer generated from retrieved context."
        )

    elif "refused" in status:

        st.warning(
            "The system could not find enough reliable "
            "context to answer this question."
        )

    else:

        st.info(
            f"Status: {status}"
        )

    # --------------------------------------------------------
    # Sources
    # --------------------------------------------------------

    st.subheader("📚 Sources")

    sources = result.get(
        "sources",
        [],
    )

    if sources:

        for index, source in enumerate(
            sources,
            start=1,
        ):

            source_name = source.get(
                "source",
                "Unknown source",
            )

            chunk_id = source.get(
                "chunk_id",
                "N/A",
            )

            score = source.get(
                "score",
                None,
            )

            with st.expander(
                f"[{index}] {source_name}"
            ):

                st.write(
                    f"**Source:** {source_name}"
                )

                st.write(
                    f"**Chunk ID:** {chunk_id}"
                )

                if score is not None:

                    st.write(
                        f"**Similarity score:** "
                        f"{score:.4f}"
                    )

    else:

        st.info(
            "No sources were returned."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Answers are generated using the retrieved knowledge "
    "base context. Check the sources to verify the answer."
)