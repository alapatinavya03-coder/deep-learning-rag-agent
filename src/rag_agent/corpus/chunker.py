"""
chunker.py
==========
Document loading and chunking pipeline.

Handles ingestion of raw files (PDF and Markdown) into structured
DocumentChunk objects ready for embedding and vector store storage.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from rag_agent.agent.state import ChunkMetadata, DocumentChunk
from rag_agent.config import Settings, get_settings
from rag_agent.vectorstore.store import VectorStoreManager


class DocumentChunker:
    """
    Loads raw documents and splits them into DocumentChunk objects.

    Supports PDF and Markdown file formats. Chunking strategy uses
    recursive character splitting with configurable chunk size and
    overlap — both are interview-defensible parameters.

    Parameters
    ----------
    settings : Settings, optional
        Application settings.

    Example
    -------
    >>> chunker = DocumentChunker()
    >>> chunks = chunker.chunk_file(
    ...     Path("data/corpus/lstm.md"),
    ...     metadata_overrides={"topic": "LSTM", "difficulty": "intermediate"}
    ... )
    >>> print(f"Produced {len(chunks)} chunks")
    """

    # Default chunking parameters — justify these in your architecture diagram.
    # chunk_size: 512 tokens balances context richness with retrieval precision.
    # chunk_overlap: 50 tokens prevents concepts that span chunk boundaries
    # from being lost entirely. A common interview question.
    DEFAULT_CHUNK_SIZE = 512
    DEFAULT_CHUNK_OVERLAP = 50

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    # -----------------------------------------------------------------------
    # Public Interface
    # -----------------------------------------------------------------------

    def chunk_file(
        self,
        file_path: Path,
        metadata_overrides: dict | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> list[DocumentChunk]:
        """Load one file and produce chunks with metadata and IDs."""
        file_path = Path(file_path)

        if not file_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")

        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError(
                "chunk_overlap must be nonnegative and smaller "
                "than chunk_size."
            )

        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            raw_chunks = self._chunk_pdf(
                file_path, chunk_size, chunk_overlap
            )
        elif suffix in {".md", ".markdown"}:
            raw_chunks = self._chunk_markdown(
                file_path, chunk_size, chunk_overlap
            )
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        chunks = []

        for raw_chunk in raw_chunks:
            text = raw_chunk["text"].strip()
            if not text:
                continue

            metadata = self._infer_metadata(
                file_path, metadata_overrides
            )

            chunks.append(
                DocumentChunk(
                    chunk_id=VectorStoreManager.generate_chunk_id(
                        metadata.source, text
                    ),
                    chunk_text=text,
                    metadata=metadata,
                )
            )

        logger.info(
            "Created {} chunks from {}", len(chunks), file_path.name
        )
        return chunks

    def chunk_files(
        self,
        file_paths: list[Path],
        metadata_overrides: dict | None = None,
    ) -> list[DocumentChunk]:
        """Process files, logging failures and continuing."""
        chunks = []

        for file_path in file_paths:
            try:
                chunks.extend(
                    self.chunk_file(
                        file_path,
                        metadata_overrides=metadata_overrides,
                    )
                )
            except Exception:
                logger.exception("Failed to chunk {}", file_path)

        return chunks

    # -----------------------------------------------------------------------
    # Format-Specific Loaders
    # -----------------------------------------------------------------------

    def _chunk_pdf(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[dict]:
        """Extract PDF text and split each page."""
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import (
            RecursiveCharacterTextSplitter,
        )

        pages = PyPDFLoader(str(file_path)).load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        documents = splitter.split_documents(pages)

        return [
            {
                "text": document.page_content,
                "page": document.metadata.get("page", 0),
            }
            for document in documents
            if document.page_content.strip()
        ]

    def _chunk_markdown(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[dict]:
        """Split Markdown at headers, then split oversized sections."""
        from langchain_text_splitters import (
            MarkdownHeaderTextSplitter,
            RecursiveCharacterTextSplitter,
        )

        text = file_path.read_text(encoding="utf-8-sig")
        if not text.strip():
            return []

        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
                ("####", "h4"),
                ("#####", "h5"),
                ("######", "h6"),
            ],
            strip_headers=False,
        )

        sections = header_splitter.split_text(text)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        documents = splitter.split_documents(sections)

        return [
            {
                "text": document.page_content,
                "header": " > ".join(
                    document.metadata[key]
                    for key in ("h1", "h2", "h3", "h4", "h5", "h6")
                    if key in document.metadata
                ),
            }
            for document in documents
            if document.page_content.strip()
        ]

    def _infer_metadata(
        self,
        file_path: Path,
        overrides: dict | None = None,
    ) -> ChunkMetadata:
        """Infer topic and difficulty from the filename."""
        import re

        stem = file_path.stem.lower()
        tokens = set(re.split(r"[^a-z0-9]+", stem))

        topic_aliases = {
            "ANN": {"ann", "mlp", "perceptron"},
            "CNN": {"cnn", "alexnet", "vgg", "resnet", "convolution"},
            "RNN": {"rnn", "recurrent"},
            "LSTM": {"lstm", "hochreiter1997"},
            "Seq2Seq": {"seq2seq"},
            "Autoencoder": {"autoencoder", "autoencoders", "vae"},
            "SOM": {"som", "kohonen"},
            "BoltzmannMachine": {"boltzmannmachine", "boltzmann", "rbm"},
            "GAN": {"gan", "gans"},
        }

        topic = "ANN"

        # Prefer the leading topic token in names such as
        # lstm_rnn_comparison_advanced.md.
        leading_token = re.split(r"[^a-z0-9]+", stem)[0]

        for candidate, aliases in topic_aliases.items():
            if leading_token in aliases:
                topic = candidate
                break
        else:
            for candidate, aliases in topic_aliases.items():
                if tokens & aliases:
                    topic = candidate
                    break
            else:
                logger.warning(
                    "Could not infer topic for {}; defaulting to ANN. "
                    "Provide a topic override if needed.",
                    file_path.name,
                )

        difficulty = "beginner"
        for candidate in ("advanced", "intermediate", "beginner"):
            if candidate in tokens:
                difficulty = candidate
                break

        values = {
            "topic": topic,
            "difficulty": difficulty,
            "type": "concept_explanation",
            "source": file_path.name,
            "related_topics": [],
        }

        values.update(overrides or {})

        if "is_bonus" not in values:
            values["is_bonus"] = values["topic"] in {
                "SOM", "BoltzmannMachine", "GAN"
            }

        return ChunkMetadata(**values)