"""
RAG service: ingest documents (chunk, embed, store in Chroma), retrieve context for LLM.
"""

from __future__ import annotations

from pathlib import Path

from sdk import get_logger, get_rag_section
from modules.rag.embed import OllamaEmbedClient
from modules.rag.store import RAGStore

logger = get_logger("rag")


class RAGService:
    """
    Facade: ingest(paths), retrieve(query) -> str, list_indexed_sources(), remove_from_index(source), clear_index().
    Uses config for embedding model, Chroma path, top_k, chunk settings.
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        self._embed = OllamaEmbedClient(
            base_url=config["base_url"],
            model_name=config["embedding_model"],
        )
        self._store = RAGStore(
            vector_db_path=config["vector_db_path"],
            embed_client=self._embed,
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
            chroma_host=config.get("chroma_host"),
            chroma_port=config.get("chroma_port"),
        )
        self._top_k = config["top_k"]
        self._document_qa_top_k = config.get("document_qa_top_k", config["top_k"])
        self._min_query_length = config.get("min_query_length", 3)

    def ingest(self, paths: list[Path]) -> None:
        """Read, chunk, embed, and store documents; replace existing chunks for same source."""
        self._store.add_documents(paths)

    def ingest_text(self, source: str, text: str) -> None:
        """Chunk, embed, and store text under the given source (e.g. stored web page). Replaces existing chunks for that source."""
        self._store.add_text(source, text)

    def retrieve(
        self, query: str, top_k: int | None = None, min_query_length: int | None = None
    ) -> str:
        """Return formatted context string for the LLM, or empty string."""
        k = top_k if top_k is not None else self._top_k
        mql = (
            min_query_length if min_query_length is not None else self._min_query_length
        )
        return self._store.retrieve(query, top_k=k, min_query_length=mql)

    def get_document_qa_top_k(self) -> int:
        return self._document_qa_top_k

    def list_indexed_sources(self) -> list[str]:
        return self._store.list_indexed_sources()

    def remove_from_index(self, source: str) -> None:
        self._store.remove_from_index(source)

    def clear_index(self) -> None:
        self._store.clear_index()

    def has_documents(self) -> bool:
        """True if the collection has at least one chunk (for empty-state check)."""
        return self._store.count() > 0


def register_with_pipeline(pipeline: object, config: object) -> RAGService | None:
    """
    Build RAGService from config, register retriever and has_documents with pipeline.
    Returns the RAGService instance for the UI (e.g. Documents dialog), or None on failure.
    If server mode is enabled, returns remote API client instead.
    """
    raw = getattr(config, "_raw", config) if config is not None else {}
    if not isinstance(raw, dict):
        raw = {}

    # Check if server mode is enabled
    from modules.api.config import get_module_server_config, get_module_base_url

    server_config = get_module_server_config(raw, "rag")
    if server_config is not None:
        # Server mode: return remote API client
        from modules.api.client import ModuleAPIClient
        from modules.api.rag_client import RemoteRAGService

        base_url = get_module_base_url(server_config)
        client = ModuleAPIClient(
            base_url=base_url,
            timeout_sec=server_config["timeout_sec"],
            retry_max=server_config["retry_max"],
            retry_delay_sec=server_config["retry_delay_sec"],
            circuit_breaker_failure_threshold=server_config[
                "circuit_breaker_failure_threshold"
            ],
            circuit_breaker_recovery_timeout_sec=server_config[
                "circuit_breaker_recovery_timeout_sec"
            ],
            api_key=server_config["api_key"],
            module_name="rag",
            use_service_discovery=server_config.get("use_service_discovery", False),
            consul_host=server_config.get("consul_host"),
            consul_port=server_config.get("consul_port", 8500),
            keydb_host=server_config.get("keydb_host"),
            keydb_port=server_config.get("keydb_port", 6379),
            load_balancing_strategy=server_config.get(
                "load_balancing_strategy", "health_based"
            ),
            health_check_interval_sec=server_config.get(
                "health_check_interval_sec", 30.0
            ),
        )
        service = RemoteRAGService(client)
    else:
        # In-process mode: return local implementation
        cfg = get_rag_section(raw)
        # Resolve *.service.consul via Consul (authoritative name server)
        if hasattr(config, "resolve_internal_service_url"):
            cfg["base_url"] = config.resolve_internal_service_url(cfg["base_url"])
            ch = cfg.get("chroma_host") or ""
            if ch and ".service.consul" in ch.lower():
                from urllib.parse import urlparse

                chroma_url = config.resolve_internal_service_url(
                    "http://" + ch + ":" + str(cfg.get("chroma_port", 8000))
                )
                resolved_host = urlparse(chroma_url).hostname
                if resolved_host:
                    cfg["chroma_host"] = resolved_host
        service = RAGService(cfg)

    def rag_retriever(query: str, top_k: int | None = None) -> str:
        return service.retrieve(query, top_k=top_k)

    pipeline.set_rag_retriever(rag_retriever)
    pipeline.set_rag_has_documents(service.has_documents)
    pipeline.set_document_qa_top_k(service.get_document_qa_top_k())
    return service


def register(context: dict) -> None:
    """
    Register RAG with the pipeline (two-phase).
    Phase 1 (context has no "pipeline"): no-op.
    Phase 2 (context has "pipeline"): call register_with_pipeline, set context["rag_service"].
    """
    pipeline = context.get("pipeline")
    if pipeline is None:
        return
    config = context.get("config")
    if config is None:
        return
    try:
        service = register_with_pipeline(pipeline, config)
        context["rag_service"] = service
    except Exception as e:
        logger.warning("RAG register failed: %s", e)
        broadcast = context.get("broadcast")
        if callable(broadcast):
            broadcast(
                {"type": "debug", "message": "[WARN] RAG not available: " + str(e)}
            )
