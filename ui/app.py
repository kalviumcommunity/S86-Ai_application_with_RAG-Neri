import json
import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🤖",
    layout="wide",
)


st.title("🤖 Grounded RAG Assistant")

st.write(
    "Ask a question and receive a grounded answer "
    "with verifiable source citations."
)


# --------------------------------------------------
# Question input
# --------------------------------------------------

question = st.text_input(
    "Ask a question",
    placeholder=(
        "What should a technician do if abnormal vibration is detected?"
    ),
)


ask_button = st.button(
    "Ask",
    type="primary",
)


# --------------------------------------------------
# Ask question
# --------------------------------------------------

if ask_button:

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

        st.stop()


    st.divider()

    st.subheader("Answer")


    answer_placeholder = st.empty()


    sources_placeholder = st.empty()


    error_placeholder = st.empty()


    answer = ""

    sources = []

    stream_error = None


    try:

        # --------------------------------------------------
        # Start streaming request
        # --------------------------------------------------

        with st.spinner("Retrieving context..."):

            response = requests.post(
                f"{API_URL}/query/stream",
                json={
                    "question": question
                },
                stream=True,
                timeout=120,
            )


        # --------------------------------------------------
        # HTTP error
        # --------------------------------------------------

        if response.status_code != 200:

            raise RuntimeError(
                f"API returned HTTP {response.status_code}"
            )


        # --------------------------------------------------
        # Read SSE events
        # --------------------------------------------------

        for raw_line in response.iter_lines(
            decode_unicode=True
        ):

            if not raw_line:
                continue


            if not raw_line.startswith("data: "):
                continue


            try:

                event = json.loads(
                    raw_line[6:]
                )

            except json.JSONDecodeError:

                continue


            event_type = event.get(
                "type"
            )


            # ----------------------------------------------
            # Citations
            # ----------------------------------------------

            if event_type == "citations":

                sources = event.get(
                    "sources",
                    []
                )


            # ----------------------------------------------
            # Token
            # ----------------------------------------------

            elif event_type == "token":

                answer += event.get(
                    "text",
                    ""
                )

                answer_placeholder.markdown(
                    answer + "▌"
                )


            # ----------------------------------------------
            # Error
            # ----------------------------------------------

            elif event_type == "refusal":

                stream_error = event.get(
                    "message",
                    "I don't have enough reliable context to answer that."
                )

                answer_placeholder.warning(
                    answer
                )

            # ----------------------------------------------
            # Done
            # ----------------------------------------------

            elif event_type == "done":

                break


        # --------------------------------------------------
        # Final answer
        # --------------------------------------------------

        if answer:

            answer_placeholder.markdown(
                answer
            )


        # --------------------------------------------------
        # Error display
        # --------------------------------------------------

        if stream_error:

            error_placeholder.error(
                stream_error
            )


        # --------------------------------------------------
        # Sources
        # --------------------------------------------------

        if sources:

            st.divider()

            st.subheader(
                "📚 Sources"
            )


            for source in sources:

                label = source.get(
                    "label",
                    "[?]"
                )

                document = source.get(
                    "document",
                    "Unknown document"
                )

                chunk_id = source.get(
                    "chunk_id",
                    "Unknown chunk"
                )

                score = source.get(
                    "score"
                )


                title = (
                    f"{label} {document} "
                    f"— {chunk_id}"
                )


                with st.expander(title):

                    if score is not None:

                        st.caption(
                            f"Similarity score: {score:.3f}"
                        )


                    if source.get(
                        "section"
                    ):

                        st.caption(
                            f"Section: "
                            f"{source['section']}"
                        )


                    st.write(
                        source.get(
                            "text",
                            "No source text available."
                        )
                    )


        elif not stream_error:

            st.info(
                "No sources were returned."
            )


    except requests.exceptions.Timeout:

        error_placeholder.error(
            "The request timed out. Please try again."
        )


    except requests.exceptions.ConnectionError:

        error_placeholder.error(
            "Could not connect to the RAG API. "
            "Make sure the backend server is running."
        )


    except Exception as error:

        error_placeholder.error(
            f"Something went wrong: {error}"
        )