"""
store.py
========
ChromaDB vector store management.

Handles all interactions with the persistent ChromaDB collection:
initialisation, ingestion, duplicate detection, and retrieval.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from loguru import logger

from rag_agent.agent.state import (
    ChunkMetadata,
    DocumentChunk,
    IngestionResult,
    RetrievedChunk,
)
from rag_agent.config import EmbeddingFactory, Settings, get_settings


class VectorStoreManager:
    """
    Manages the ChromaDB persistent vector store for the corpus.

    All corpus ingestion and retrieval operations pass through this class.
    It is the single point of contact between the application and ChromaDB.

    Parameters
    ----------
    settings : Settings, optional
        Application settings. Uses get_settings() singleton if not provided.

    Example
    -------
    >>> manager = VectorStoreManager()
    >>> result = manager.ingest(chunks)
    >>> print(f"Ingested: {result.ingested}, Skipped: {result.skipped}")
    >>>
    >>> chunks = manager.query("explain the vanishing gradient problem", k=4)
    >>> for chunk in chunks:
    ...     print(chunk.to_citation(), chunk.score)
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._embeddings = EmbeddingFactory(self._settings).create()
        self._client = None
        self._collection = None
        self._initialise()

    # -----------------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------------

    def _initialise(self) -> None:
        """Connect to the persistent ChromaDB collection."""
        import chromadb

        try:
            db_path = Path(self._settings.chroma_db_path)
            db_path.mkdir(parents=True, exist_ok=True)

            self._client = chromadb.PersistentClient(
                path=str(db_path)
            )

            self._collection = self._client.get_or_create_collection(
                name=self._settings.chroma_collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            logger.info(
                "Collection '{}' initialized with {} chunks",
                self._settings.chroma_collection_name,
                self._collection.count(),
            )

        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize ChromaDB at "
                f"{self._settings.chroma_db_path}"
            ) from exc

    # -----------------------------------------------------------------------
    # Duplicate Detection
    # -----------------------------------------------------------------------

    @staticmethod
    def generate_chunk_id(source: str, chunk_text: str) -> str:
        """
        Generate a deterministic chunk ID from source filename and content.

        Using a content hash ensures two uploads of the same file produce
        the same IDs, making duplicate detection reliable regardless of
        filename changes.

        Parameters
        ----------
        source : str
            The source filename (e.g. 'lstm.md').
        chunk_text : str
            The full text content of the chunk.

        Returns
        -------
        str
            A 16-character hex string derived from SHA-256 of the inputs.
        """
        content = f"{source}::{chunk_text}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def check_duplicate(self, chunk_id: str) -> bool:
        result = self._collection.get(
            ids=[chunk_id],
            include=[],
        )
        return chunk_id in result["ids"]

    # -----------------------------------------------------------------------
    # Ingestion
    # -----------------------------------------------------------------------

    def ingest(self, chunks: list[DocumentChunk]) -> IngestionResult:
        """Embed and store new chunks, skipping duplicates."""
        result = IngestionResult()
        seen_ids: set[str] = set()
        pending: list[DocumentChunk] = []

        for chunk in chunks:
            try:
                if (
                    chunk.chunk_id in seen_ids
                    or self.check_duplicate(chunk.chunk_id)
                ):
                    result.skipped += 1
                    continue

                if not chunk.chunk_text.strip():
                    raise ValueError("Chunk text is empty.")

                seen_ids.add(chunk.chunk_id)
                pending.append(chunk)

            except Exception as exc:
                result.errors.append(f"{chunk.chunk_id}: {exc}")

        batch_size = self._settings.ingestion_batch_size

        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]

            try:
                texts = [chunk.chunk_text for chunk in batch]
                embeddings = self._embeddings.embed_documents(texts)

                self._collection.upsert(
                    ids=[chunk.chunk_id for chunk in batch],
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=[
                        chunk.metadata.to_dict() for chunk in batch
                    ],
                )

                result.ingested += len(batch)

                for chunk in batch:
                    source = chunk.metadata.source
                    if source not in result.document_ids:
                        result.document_ids.append(source)

            except Exception as exc:
                for chunk in batch:
                    result.errors.append(f"{chunk.chunk_id}: {exc}")
                logger.exception("Failed to ingest a chunk batch")

        logger.info(
            "Ingestion: {} ingested, {} skipped, {} errors",
            result.ingested,
            result.skipped,
            len(result.errors),
        )
        return result
    # -----------------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        k: int | None = None,
        topic_filter: str | None = None,
        difficulty_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve relevant chunks with optional metadata filters."""
        if not query_text.strip():
            return []

        k = self._settings.retrieval_k if k is None else k
        if k <= 0:
            raise ValueError("k must be greater than zero.")

        total_chunks = self._collection.count()
        if total_chunks == 0:
            return []

        filters = []

        if topic_filter:
            filters.append({"topic": topic_filter})

        if difficulty_filter:
            filters.append({"difficulty": difficulty_filter})

        where_filter = None
        if len(filters) == 1:
            where_filter = filters[0]
        elif len(filters) > 1:
            where_filter = {"$and": filters}

        query_embedding = self._embeddings.embed_query(query_text)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, total_chunks),
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        retrieved: list[RetrievedChunk] = []

        for chunk_id, text, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            score = 1.0 - float(distance)

            if score < self._settings.similarity_threshold:
                continue

            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    chunk_text=text,
                    metadata=ChunkMetadata.from_dict(metadata),
                    score=score,
                )
            )

        return sorted(
            retrieved,
            key=lambda chunk: chunk.score,
            reverse=True,
        )

    # -----------------------------------------------------------------------
    # Corpus Inspection
    # -----------------------------------------------------------------------

    def list_documents(self) -> list[dict]:
        """List source documents and their chunk counts."""
        results = self._collection.get(include=["metadatas"])
        documents = {}

        for metadata in results["metadatas"] or []:
            if not metadata:
                continue

            source = metadata.get("source")
            if not source:
                continue

            if source not in documents:
                documents[source] = {
                    "source": source,
                    "topic": metadata.get("topic", "unknown"),
                    "chunk_count": 0,
                }

            documents[source]["chunk_count"] += 1

        return [
            documents[source]
            for source in sorted(documents)
        ]

    def get_document_chunks(self, source: str) -> list[DocumentChunk]:
        """
        Retrieve all chunks belonging to a specific source document.

        Used by the document viewer to display document content.

        Parameters
        ----------
        source : str
            The source filename to retrieve chunks for.

        Returns
        -------
        list[DocumentChunk]
            All chunks from this source, ordered by their position
            in the original document.
        """
        # TODO: implement
        # self._collection.get(where={"source": source}, include=["documents", "metadatas"])
        # Reconstruct DocumentChunk objects from results
        raise NotImplementedError

    def get_collection_stats(self) -> dict:
        """
        Return summary statistics about the current collection.

        Used by the UI to show corpus health at a glance.

        Returns
        -------
        dict
            Keys: total_chunks, topics (list), sources (list),
            bonus_topics_present (bool).
        """
        # TODO: implement
        raise NotImplementedError

    def delete_document(self, source: str) -> int:
        """
        Remove all chunks from a specific source document.

        Parameters
        ----------
        source : str
            Source filename to remove.

        Returns
        -------
        int
            Number of chunks deleted.
        """
        # TODO: implement
        # self._collection.delete(where={"source": source})
        raise NotImplementedError
