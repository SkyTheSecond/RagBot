# RAG Knowledge Assistant

A retrieval-augmented generation (RAG) system for chatting with your own documents. Upload PDFs, Markdown, or text files, and ask natural-language questions that are answered using their content — with source attribution, a groundedness/confidence score, and an agent that can pull more context on its own when the initial search results aren't enough.

## How It Works

```
Upload (.pdf / .md / .txt)
        │
        ▼
DocumentLoader            heading detection + semantic chunking
        │
        ├──► MongoDB      full document text, heading chunks, semantic chunks
        └──► Qdrant       semantic chunks, embedded as vectors

Question
        │
        ▼
RAGPipeline (LangGraph agent)
   retrieve → generate ⇄ execute_tool (Mongo lookups, up to 3 loops) → grade → answer
```

**Ingestion.** Each document is parsed and split into heading-based sections (using ATX headers for Markdown, font-size/weight heuristics for PDFs, and ALL-CAPS/underline heuristics for plain text), then semantically chunked within those sections using LangChain's embedding-based `SemanticChunker` rather than fixed-size splitting. Every chunk keeps a pointer to its parent section and to its previous/next neighbor, so the model can later ask for "more context around this" without re-running search. The full text and both chunk levels go to MongoDB; the semantic chunks are embedded and stored in Qdrant for vector search.

**Answering.** A question first gets rewritten into a self-contained search query if there's prior conversation history (so a follow-up like "what about its capital?" resolves the pronoun to the actual entity from the last answer). The rewritten query is embedded and searched against Qdrant for the top-k chunks. Those chunks are handed to the LLM along with three tools backed by MongoDB — fetch surrounding chunks, fetch a whole section, or fetch the entire document — so it can request more context when the initial excerpts aren't sufficient, in increasing order of cost. Once the model produces a final answer, a separate grading step scores how well that answer is actually grounded in the retrieved/tool-fetched material, and that score is what surfaces as the confidence indicator in the UI.

## Tech Stack

| Layer | Technology |
|---|---|
| API server | FastAPI |
| LLM (chat, generation, tool calls) | Groq (cloud) |
| Embeddings | HuggingFace `sentence-transformers` (local) |
| Vector store | Qdrant |
| Document / chunk store | MongoDB |
| Chunking | LangChain `SemanticChunker` |
| Agent orchestration | LangGraph |
| PDF parsing | PyMuPDF (`fitz`) |
| Frontend | React + TypeScript + Vite + Tailwind CSS |
| Markdown rendering | `react-markdown`, `remark-gfm`, `remark-breaks` |
| Icons | `lucide-react` |

## Project Structure

```
backend/
├── server.py              FastAPI app — routes, startup/shutdown, session cache
├── config.py               Environment-driven settings (Groq, Qdrant, Mongo, HF)
├── inference_client.py     Unified client: Groq for chat, HuggingFace for embeddings
├── document_loader.py      PDF/MD/TXT parsing, heading detection, semantic chunking
├── ingestion_pipeline.py   Orchestrates writes/deletes across Mongo + Qdrant
├── mongo_store.py          MongoDB wrapper (documents, heading chunks, semantic chunks)
├── qdrant_store.py         Qdrant wrapper (embedding, upsert, similarity search)
├── chroma_store.py         Alternate vector store for local-only deployments
└── new_test.py / test.py   Integration-style test scripts

frontend/
└── src/
    ├── main.tsx             Router setup (/chat, /dashboard)
    ├── index.css             Tailwind entry + global styles
    ├── pages/
    │   ├── Chat.tsx          Chat interface
    │   └── Dashboard.tsx     Document management UI
    └── lib/
        ├── api.ts             Typed fetch wrappers for the backend API
        └── markdown.tsx       Styled Markdown renderer for chat answers
```

## Setup

### Prerequisites

- Python 3.11+
- Node.js 18+
- A running MongoDB instance
- A running Qdrant instance (local or [Qdrant Cloud](https://cloud.qdrant.io))
- A [Groq API key](https://console.groq.com)

### Backend

```bash
cd backend
uv sync
```

Create a `.env` file in the backend directory (see [Environment Variables](#environment-variables) below), then run:

```bash
uv run uvicorn server:app --host 0.0.0.0 --port 8080 --reload
```

### Frontend

```bash
cd frontend
pnpm install
pnpm run dev
```

The frontend calls the API at relative paths under `/v1` (see `lib/api.ts`), so point your dev server's proxy (e.g. Vite's `server.proxy`) at `http://localhost:8080`, or serve both behind the same origin in production.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | API key for Groq Cloud (LLM inference) |
| `GROQ_MODEL` | `meta-llama/llama-4-scout-17b-16e-instruct` | Groq model used for chat/generation |
| `HF_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model |
| `HF_DEVICE` | `cpu` | Device for the embedding model (`cpu` / `cuda`) |
| `QDRANT_HOST` | `localhost` | Qdrant host (or Qdrant Cloud URL) |
| `QDRANT_PORT` | `6333` | Qdrant port |
| `QDRANT_COLLECTION` | `rag_chunks` | Qdrant collection name |
| `QDRANT_API_KEY` | — | Required for Qdrant Cloud |
| `MONGO_HOST` | `localhost` | MongoDB host |
| `MONGO_PORT` | `27017` | MongoDB port |
| `MONGO_USER` | `admin` | MongoDB username |
| `MONGO_PASSWORD` | `password` | MongoDB password |
| `MONGO_DB` | `rag` | MongoDB database name |
| `MONGO_URI` | built from the above | Overrides the auto-built connection string if set |

`config.py` also defines `OLLAMA_*` and `CHROMA_*` variables for a fully local deployment path (local LLM + embeddings via Ollama, local vector store via Chroma). These were used early in development and are kept in place for anyone who wants to self-host without cloud dependencies later, but the current pipeline runs on Groq + Qdrant and doesn't read them.

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/v1/health` | Mongo / Qdrant / embedding model health check |
| `POST` | `/v1/chat` | Ask a question; returns answer, sources, confidence score |
| `POST` | `/v1/chat/stream` | Same as above, streamed as SSE |
| `DELETE` | `/v1/chat/{session_id}` | Clear a session's conversation history |
| `GET` | `/v1/documents` | List ingested documents |
| `POST` | `/v1/documents/upload` | Upload and ingest a document (`.pdf`, `.md`, `.txt`) |
| `DELETE` | `/v1/documents/{doc_id}` | Delete a document and its chunks from both stores |
| `POST` | `/v1/documents/{doc_id}/reingest` | Replace a document with a new version |

## Testing

`new_test.py` is the primary test script — it exercises every layer (inference client, document loader, Mongo, Qdrant, ingestion pipeline, and a full end-to-end RAG run) against live services, reporting all checks rather than stopping at the first failure:

```bash
uv run python new_test.py
```

It requires a reachable Groq API key, MongoDB, and Qdrant instance, since these are integration tests rather than mocked unit tests.
