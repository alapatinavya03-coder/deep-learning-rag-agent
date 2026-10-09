"""
app.py
======
Streamlit user interface for the Deep Learning RAG Interview Prep Agent.

Three-panel layout:
  - Left sidebar: Document ingestion and corpus browser
  - Centre: Document viewer
  - Right: Chat interface

API contract with the backend (agree this with Pipeline Engineer
before building anything):

  ingest(file_paths: list[Path]) -> IngestionResult
  list_documents() -> list[dict]
  get_document_chunks(source: str) -> list[DocumentChunk]
  chat(query: str, history: list[dict], filters: dict) -> AgentResponse

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from rag_agent.agent.graph import get_compiled_graph
from rag_agent.agent.state import AgentResponse
from rag_agent.config import get_settings
from rag_agent.corpus.chunker import DocumentChunker
from rag_agent.vectorstore.store import VectorStoreManager
from tempfile import TemporaryDirectory
from uuid import uuid4

from langchain_core.messages import HumanMessage

from rag_agent.agent.state import IngestionResult
from rag_agent.config import LLMFactory
from rag_agent.vectorstore.store import VectorStoreManager


# ---------------------------------------------------------------------------
# Cached Resources
# ---------------------------------------------------------------------------
# Use st.cache_resource for objects that should persist across reruns
# and be shared across all user sessions. This prevents re-initialising
# ChromaDB and reloading the embedding model on every button click.


@st.cache_resource
def get_llm():
    """Create the LLM once and reuse it across reruns."""
    return LLMFactory().create()


@st.cache_resource
def get_chunker() -> DocumentChunker:
    """Return the singleton DocumentChunker."""
    return DocumentChunker()


@st.cache_resource
def get_graph():
    """Return the compiled LangGraph agent."""
    return get_compiled_graph()

@st.cache_resource
def get_vector_store() -> VectorStoreManager:
    return VectorStoreManager()
# ---------------------------------------------------------------------------
# Session State Initialisation
# ---------------------------------------------------------------------------


def initialise_session_state() -> None:
    """
    Initialise all st.session_state keys on first run.

    Must be called at the top of main() before any UI is rendered.
    Without this, state keys referenced in callbacks will raise KeyError.

    Interview talking point: Streamlit reruns the entire script on every
    user interaction. session_state is the mechanism for persisting data
    (chat history, ingestion results) across reruns.
    """
    defaults = {
        "chat_history": [],           # list of {"role": "user"|"assistant", "content": str}
        "ingested_documents": [],     # list of dicts from list_documents()
        "selected_document": None,    # source filename currently in viewer
        "last_ingestion_result": None,
        "thread_id": str(uuid4()),  # LangGraph conversation thread
        "topic_filter": None,
        "difficulty_filter": None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


# ---------------------------------------------------------------------------
# Ingestion Panel (Sidebar)
# ---------------------------------------------------------------------------


def render_ingestion_panel(
    store: VectorStoreManager,
    chunker: DocumentChunker,
) -> None:
    """Upload, chunk, and ingest study materials."""
    st.sidebar.header("📂 Corpus Ingestion")

    uploads = st.sidebar.file_uploader(
        "Upload study materials",
        type=["pdf", "md"],
        accept_multiple_files=True,
    )

    if st.sidebar.button(
        "Ingest Documents",
        disabled=not uploads,
    ):
        result = IngestionResult()

        with st.spinner("Processing documents..."):
            with TemporaryDirectory() as temp_dir:
                for index, upload in enumerate(uploads):
                    try:
                        # Separate folders prevent filename collisions.
                        folder = Path(temp_dir) / str(index)
                        folder.mkdir()

                        name = upload.name.replace("\\", "/").split("/")[-1]
                        file_path = folder / name
                        file_path.write_bytes(upload.getvalue())

                        chunks = chunker.chunk_file(file_path)

                        if not chunks:
                            result.errors.append(
                                f"{name}: No readable text found."
                            )
                            continue

                        file_result = store.ingest(chunks)
                        result.ingested += file_result.ingested
                        result.skipped += file_result.skipped
                        result.errors.extend(file_result.errors)

                        for source in file_result.document_ids:
                            if source not in result.document_ids:
                                result.document_ids.append(source)

                    except Exception as exc:
                        result.errors.append(f"{upload.name}: {exc}")

        st.session_state.last_ingestion_result = result

    result = st.session_state.last_ingestion_result

    if result is not None:
        summary = (
            f"{result.ingested} chunks added, "
            f"{result.skipped} duplicates skipped"
        )

        if result.errors:
            st.sidebar.warning(summary)
            for error in result.errors:
                st.sidebar.error(error)
        elif result.ingested:
            st.sidebar.success(summary)
        else:
            st.sidebar.info(summary)

    documents = store.list_documents()
    st.session_state.ingested_documents = documents

    st.sidebar.subheader("Stored Documents")

    for document in documents:
        source = document["source"]

        st.sidebar.write(
            f"{source} — {document['topic']} "
            f"({document['chunk_count']} chunks)"
        )

        if st.sidebar.button(
            "🗑 Remove",
            key=f"remove_document_{source}",
        ):
            store.delete_document(source)
            st.rerun()


def render_corpus_stats(store: VectorStoreManager) -> None:
    """
    Render a compact corpus health summary in the sidebar.

    Shows total chunks, topics covered, and whether bonus topics
    are present. Used during Hour 3 to demonstrate corpus completeness.

    Parameters
    ----------
    store : VectorStoreManager
    """
    # TODO: implement
    # stats = store.get_collection_stats()
    # st.sidebar.metric("Total Chunks", stats["total_chunks"])
    # st.sidebar.write("Topics:", ", ".join(stats["topics"]))
    # if stats["bonus_topics_present"]:
    #     st.sidebar.success("✅ Bonus topics present")
    # else:
    #     st.sidebar.warning("⚠️ No bonus topics yet")
    pass


# ---------------------------------------------------------------------------
# Document Viewer Panel (Centre)
# ---------------------------------------------------------------------------


def render_document_viewer(store: VectorStoreManager) -> None:
    """
    Render the document viewer in the main centre column.

    Displays a selectable list of ingested documents. When a document
    is selected, renders its chunk content in a scrollable pane.

    Parameters
    ----------
    store : VectorStoreManager
    """
    st.subheader("📄 Document Viewer")

    # TODO: implement
    # 1. If no documents ingested: show placeholder message
    #
    # 2. st.selectbox("Select document", options=[doc["source"] for doc in docs])
    #    Store selection in st.session_state["selected_document"]
    #
    # 3. On selection change: store.get_document_chunks(selected_source)
    #
    # 4. Render chunks in a scrollable container (st.container with fixed height)
    #    For each chunk:
    #    - Show metadata badge: topic | difficulty | type
    #    - Show chunk text
    #    - Show similarity score if this chunk was used in last response
    #
    # 5. Display chunk count and coverage summary below viewer

    st.info("Ingest documents using the sidebar to view content here.")


# ---------------------------------------------------------------------------
# Chat Interface Panel (Right)
# ---------------------------------------------------------------------------


def render_chat_interface(graph) -> None:
    """Display chat and submit questions to the RAG graph."""
    st.subheader("💬 Interview Prep Chat")

    col_topic, col_diff = st.columns(2)

    with col_topic:
        topic = st.selectbox(
            "Topic",
            [
                "All", "ANN", "CNN", "RNN", "LSTM", "Seq2Seq",
                "Autoencoder", "SOM", "BoltzmannMachine", "GAN",
            ],
        )

    with col_diff:
        difficulty = st.selectbox(
            "Difficulty",
            ["All", "beginner", "intermediate", "advanced"],
        )

    st.session_state.topic_filter = None if topic == "All" else topic
    st.session_state.difficulty_filter = (
        None if difficulty == "All" else difficulty
    )

    chat_container = st.container(height=400)

    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                if message.get("sources"):
                    with st.expander("📎 Sources"):
                        for source in message["sources"]:
                            st.caption(source)

                if message.get("no_context_found"):
                    st.warning("No relevant content found in corpus.")

    query = st.chat_input("Ask about a deep learning topic...")

    if query:
        st.session_state.chat_history.append(
            {"role": "user", "content": query}
        )

        with chat_container:
            with st.chat_message("user"):
                st.markdown(query)

        try:
            with st.spinner("Searching your documents..."):
                result = graph.invoke(
                    {
                        "messages": [HumanMessage(content=query)],
                        "original_query": query,
                        "rewritten_query": "",
                        "retrieved_chunks": [],
                        "no_context_found": False,
                        "final_response": None,
                        "topic_filter": st.session_state.topic_filter,
                        "difficulty_filter": (
                            st.session_state.difficulty_filter
                        ),
                    },
                    config={
                        "configurable": {
                            "thread_id": st.session_state.thread_id
                        }
                    },
                )

                response = result.get("final_response")

                if not isinstance(response, AgentResponse):
                    raise RuntimeError(
                        "The graph did not return an AgentResponse."
                    )

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": response.answer,
                    "sources": response.sources,
                    "no_context_found": response.no_context_found,
                }
            )

        except Exception as exc:
            st.error(f"Could not generate an answer: {exc}")
            return

        st.rerun()

# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Application entry point.

    Sets page config, initialises session state, instantiates shared
    resources, and renders all UI panels.

    Run with: uv run streamlit run src/rag_agent/ui/app.py
    """
    settings = get_settings()

    st.set_page_config(
        page_title=settings.app_title,
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title(f"🧠 {settings.app_title}")
    st.caption(
        "RAG-powered interview preparation — built with LangChain, LangGraph, and ChromaDB"
    )

    initialise_session_state()

    # Instantiate shared backend resources
    store = get_vector_store()
    chunker = get_chunker()
    graph = get_graph()

    # Sidebar
    render_ingestion_panel(store, chunker)
    render_corpus_stats(store)

    # Main content area — two columns
    viewer_col, chat_col = st.columns([1, 1], gap="large")

    with viewer_col:
        render_document_viewer(store)

    with chat_col:
        render_chat_interface(graph)


if __name__ == "__main__":
    main()
