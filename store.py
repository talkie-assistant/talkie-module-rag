"""
Chroma vector store for RAG: add documents (replace-by-source), retrieve with source attribution.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from modules.rag.chunk import chunk_text
from modules.rag.embed import OllamaEmbedClient
from modules.rag.pdf import extract_text_from_pdf

logger = logging.getLogger(__name__)

COLLECTION_NAME = "talkie_docs"


def _read_file_text(path: Path) -> str:
    """Read full text from .txt or .pdf."""
    suf = path.suffix.lower()
    if suf == ".txt":
        return path.read_text(encoding="utf-8", errors="replace")
    if suf == ".pdf":
        return extract_text_from_pdf(path)
    return ""


class RAGStore:
    """Chroma-backed store: add_documents (replace by source), retrieve, list_indexed_sources, remove_from_index, clear_index.
    Uses chromadb.HttpClient when chroma_host (and optionally chroma_port) are set; otherwise chromadb.PersistentClient.
    """

    def __init__(
        self,
        vector_db_path: str,
        embed_client: OllamaEmbedClient,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        chroma_host: str | None = None,
        chroma_port: int | None = None,
    ) -> None:
        self._path = vector_db_path
        self._embed = embed_client
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        host = (chroma_host or "").strip()
        port = chroma_port if chroma_port is not None else 8000
        if host:
            self._client = chromadb.HttpClient(
                host=host,
                port=port,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            logger.info("RAGStore using Chroma HTTP server %s:%s", host, port)
        else:
            Path(vector_db_path).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=vector_db_path,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "Talkie RAG documents"},
        )

    def add_documents(self, paths: list[Path]) -> None:
        """
        For each path: read text, chunk, embed, add to Chroma.
        Before adding a file's chunks, delete existing chunks with same source (filename).
        """
        for path in paths:
            if not path.is_file():
                logger.warning("Skipping non-file %s", path)
                continue
            source_name = path.name
            try:
                text = _read_file_text(path)
            except Exception as e:
                logger.exception("Read failed for %s: %s", path, e)
                raise
            if not text or not text.strip():
                logger.warning("Empty text for %s", path)
                continue
            chunks = chunk_text(text, self._chunk_size, self._chunk_overlap)
            if not chunks:
                continue
            # Remove existing chunks for this source (re-vectorize = replace)
            try:
                self._collection.delete(where={"source": source_name})
            except Exception as e:
                logger.debug("Delete by source (may be none): %s", e)
            ids = [f"{path.stem}_{i}" for i in range(len(chunks))]
            metadatas = [
                {"source": source_name, "chunk_index": i} for i in range(len(chunks))
            ]
            try:
                embeddings = self._embed.embed(chunks)
            except ValueError:
                raise
            if len(embeddings) != len(chunks):
                raise RuntimeError("Embed count mismatch")
            self._collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
            )
            logger.info("Indexed %s (%d chunks)", source_name, len(chunks))

    def add_text(self, source: str, text: str) -> None:
        """
        Chunk, embed, and add text under the given source (e.g. URL or label).
        Replaces existing chunks with the same source. Used for stored web pages.
        """
        source_name = (source or "").strip()
        if not source_name:
            raise ValueError("source is required")
        if not text or not text.strip():
            logger.warning("Empty text for source %s", source_name)
            return
        chunks = chunk_text(text, self._chunk_size, self._chunk_overlap)
        if not chunks:
            return
        slug = re.sub(r"[^a-zA-Z0-9_-]", "_", source_name)[:80]
        if not slug:
            slug = "web"
        try:
            self._collection.delete(where={"source": source_name})
        except Exception as e:
            logger.debug("Delete by source (may be none): %s", e)
        ids = [f"{slug}_{i}" for i in range(len(chunks))]
        metadatas = [
            {"source": source_name, "chunk_index": i} for i in range(len(chunks))
        ]
        try:
            embeddings = self._embed.embed(chunks)
        except ValueError:
            raise
        if len(embeddings) != len(chunks):
            raise RuntimeError("Embed count mismatch")
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )
        logger.info("Indexed text source %s (%d chunks)", source_name, len(chunks))

    def retrieve(self, query: str, top_k: int, min_query_length: int = 3) -> str:
        """
        Embed query, search Chroma, return formatted context string with "Source: filename" per chunk.
        Returns "" if query too short, collection empty, or on error.
        """
        if not query or len(query.strip()) < min_query_length:
            return ""
        try:
            count = self._collection.count()
            if count == 0:
                return ""
        except Exception:
            return ""
        try:
            q_embs = self._embed.embed(query.strip())
            if not q_embs:
                return ""
            results = self._collection.query(
                query_embeddings=q_embs,
                n_results=min(top_k, count),
                include=["documents", "metadatas"],
            )
        except Exception as e:
            logger.exception("Retrieve failed: %s", e)
            return ""
        docs = results.get("documents")
        metadatas = results.get("metadatas")
        if not docs or not docs[0]:
            return ""
        parts: list[str] = []
        for i, doc in enumerate(docs[0]):
            meta = (metadatas[0][i] or {}) if metadatas and metadatas[0] else {}
            source = meta.get("source", "unknown")
            parts.append(f"Source: {source}\n{doc}")
        return "\n\n".join(parts)

    def list_indexed_sources(self) -> list[str]:
        """Return unique source (filename) values in the collection."""
        try:
            data = self._collection.get(include=["metadatas"])
            metas = data.get("metadatas") or []
            seen: set[str] = set()
            for m in metas:
                if isinstance(m, dict) and m.get("source"):
                    seen.add(str(m["source"]))
            return sorted(seen)
        except Exception as e:
            logger.exception("list_indexed_sources failed: %s", e)
            return []

    def remove_from_index(self, source: str) -> None:
        """Delete all chunks with metadata source equal to the given filename."""
        try:
            self._collection.delete(where={"source": source})
            logger.info("Removed source %s from index", source)
        except Exception as e:
            logger.exception("remove_from_index failed: %s", e)
            raise

    def clear_index(self) -> None:
        """Delete all documents in the collection (reset)."""
        try:
            # Chroma: get all ids then delete
            data = self._collection.get(include=[])
            ids = data.get("ids") or []
            if ids:
                self._collection.delete(ids=ids)
            logger.info("Cleared RAG index")
        except Exception as e:
            logger.exception("clear_index failed: %s", e)
            raise

    def count(self) -> int:
        """Number of chunks in the collection (for fast empty check)."""
        try:
            return self._collection.count()
        except Exception:
            return 0
