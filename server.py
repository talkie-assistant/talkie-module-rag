"""
RAG module HTTP server.
Exposes document ingestion and retrieval via REST API.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from fastapi import Request, status

from sdk import get_logger, get_rag_section
from modules.api.server import BaseModuleServer
from modules.rag import RAGService

logger = get_logger("rag")


class RAGModuleServer(BaseModuleServer):
    """HTTP server for RAG module."""

    def __init__(
        self,
        config: dict[str, Any],
        host: str = "localhost",
        port: int = 8002,
        api_key: str | None = None,
    ) -> None:
        super().__init__(
            module_name="rag",
            module_version="1.0.0",
            host=host,
            port=port,
            api_key=api_key,
        )
        self._config = config
        self._service: RAGService | None = None
        self._setup_endpoints()

    def _setup_endpoints(self) -> None:
        """Set up RAG-specific endpoints."""

        @self._app.post("/ingest")
        async def ingest(request: Request) -> dict[str, Any]:
            """Ingest documents."""
            try:
                if r := self._require_service(self._service):
                    return r
                data = await request.json()
                paths = [Path(p) for p in data.get("paths", [])]
                self._service.ingest(paths)
                return {"success": True, "ingested_count": len(paths)}
            except Exception as e:
                logger.exception("RAG ingest failed: %s", e)
                return self._error_response(
                    status.HTTP_500_INTERNAL_SERVER_ERROR, "internal_error", str(e)
                )

        @self._app.post("/ingest_text")
        async def ingest_text(request: Request) -> dict[str, Any]:
            """Ingest text."""
            try:
                if r := self._require_service(self._service):
                    return r
                data = await request.json()
                source = data.get("source", "")
                text = data.get("text", "")
                if not source or not text:
                    return self._error_response(
                        status.HTTP_400_BAD_REQUEST,
                        "invalid_request",
                        "source and text required",
                    )
                self._service.ingest_text(source, text)
                return {"success": True}
            except Exception as e:
                logger.exception("RAG ingest_text failed: %s", e)
                return self._error_response(
                    status.HTTP_500_INTERNAL_SERVER_ERROR, "internal_error", str(e)
                )

        @self._app.post("/retrieve")
        async def retrieve(request: Request) -> dict[str, Any]:
            """Retrieve context."""
            try:
                if r := self._require_service(self._service):
                    return r
                data = await request.json()
                query = data.get("query", "")
                top_k = data.get("top_k")
                min_query_length = data.get("min_query_length")
                context = self._service.retrieve(
                    query, top_k=top_k, min_query_length=min_query_length
                )
                return {"context": context}
            except Exception as e:
                logger.exception("RAG retrieve failed: %s", e)
                return self._error_response(
                    status.HTTP_500_INTERNAL_SERVER_ERROR, "internal_error", str(e)
                )

        @self._app.get("/sources")
        async def list_sources() -> dict[str, Any]:
            """List indexed sources."""
            try:
                if r := self._require_service(self._service):
                    return r
                sources = self._service.list_indexed_sources()
                return {"sources": sources}
            except Exception as e:
                logger.exception("RAG list_sources failed: %s", e)
                return self._error_response(
                    status.HTTP_500_INTERNAL_SERVER_ERROR, "internal_error", str(e)
                )

        @self._app.delete("/sources/{source}")
        async def remove_source(source: str) -> dict[str, Any]:
            """Remove source from index."""
            try:
                if r := self._require_service(self._service):
                    return r
                self._service.remove_from_index(source)
                return {"success": True}
            except Exception as e:
                logger.exception("RAG remove_source failed: %s", e)
                return self._error_response(
                    status.HTTP_500_INTERNAL_SERVER_ERROR, "internal_error", str(e)
                )

        @self._app.post("/clear")
        async def clear() -> dict[str, Any]:
            """Clear entire index."""
            try:
                if r := self._require_service(self._service):
                    return r
                self._service.clear_index()
                return {"success": True}
            except Exception as e:
                logger.exception("RAG clear failed: %s", e)
                return self._error_response(
                    status.HTTP_500_INTERNAL_SERVER_ERROR, "internal_error", str(e)
                )

        @self._app.get("/has_documents")
        async def has_documents() -> dict[str, Any]:
            """Check if documents exist."""
            try:
                if r := self._require_service(self._service):
                    return r
                has_docs = self._service.has_documents()
                return {"has_documents": has_docs}
            except Exception as e:
                logger.exception("RAG has_documents failed: %s", e)
                return self._error_response(
                    status.HTTP_500_INTERNAL_SERVER_ERROR, "internal_error", str(e)
                )

    async def startup(self) -> None:
        """Initialize RAG service on startup."""
        await super().startup()
        try:
            rag_config = get_rag_section(self._config)
            self._service = RAGService(rag_config)
            self.set_ready(True)
            logger.info("RAG module initialized and ready")
        except Exception as e:
            logger.exception("Failed to initialize RAG module: %s", e)
            self.set_ready(False)

    async def shutdown(self) -> None:
        """Cleanup on shutdown."""
        await super().shutdown()

    def get_config_dict(self) -> dict[str, Any]:
        """Get current configuration."""
        return self._config

    def update_config_dict(self, config: dict[str, Any]) -> None:
        """Update configuration."""
        self._config.update(config)
        # Recreate service
        try:
            rag_config = get_rag_section(self._config)
            self._service = RAGService(rag_config)
        except Exception as e:
            logger.exception("Failed to update RAG config: %s", e)

    def reload_config_from_file(self) -> None:
        """Reload configuration from file; base loads config, we apply via update_config_dict."""
        try:
            super().reload_config_from_file()
        except Exception as e:
            logger.exception("Failed to reload RAG config from file: %s", e)
            raise


def main() -> None:
    """CLI entry point for RAG module server."""
    parser = argparse.ArgumentParser(description="RAG module HTTP server")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8002, help="Port to bind to")
    parser.add_argument("--api-key", help="Optional API key for authentication")
    args = parser.parse_args()

    # Load config
    from config import load_config

    config = load_config()

    # Create and run server
    server = RAGModuleServer(
        config=config,
        host=args.host,
        port=args.port,
        api_key=args.api_key,
    )
    server.run()


if __name__ == "__main__":
    main()
