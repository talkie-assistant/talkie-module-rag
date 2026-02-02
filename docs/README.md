# RAG module

Document ingest, embedding (Ollama), and retrieval for LLM context. Add PDFs and text files; ask questions by voice and get answers from your documents.

## Features

- **Ingest**: Upload PDF or text files; chunk, embed (nomic-embed-text via Ollama), store in Chroma.
- **Retrieve**: Query by voice; top-k chunks are retrieved and formatted as context for the LLM.
- **Documents UI**: List sources, remove, clear. Module config is merged with main config.

## Config

Module config is in `config.yaml` in this directory; merged with root config. Keys:

- **rag.embedding_model**: Ollama embedding model (e.g. `nomic-embed-text`).
- **rag.vector_db_path**: Chroma path (embedded) or **rag.chroma_host** / **rag.chroma_port** for server.
- **rag.top_k**, **rag.document_qa_top_k**: Number of chunks to retrieve.
- **rag.chunk_size**, **rag.chunk_overlap**: Chunking for ingest.
- **rag.min_query_length**: Minimum query length to run retrieval.

Override in root `config.yaml` or `config.user.yaml`.

## Quick reference (H key)

- **RAG mode**: When RAG is the active module, press **H** or **h** to show this help.
- **Documents**: Use the Documents panel to upload files, refresh the list, and remove sources.
