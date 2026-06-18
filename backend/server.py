# %% Imports
from __future__ import annotations

import os
import shutil
import uuid
import tempfile
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator
from collections import defaultdict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from qdrant_store import QdrantStore
from config import QdrantConfig, MongoConfig, GroqConfig, HuggingFaceConfig

from document_loader import DocumentLoader
from ingestion_pipeline import IngestionPipeline
from mongo_store import MongoStore
from inference_client import InferenceClient
from rag_pipeline import RAGPipeline

from cachetools import TTLCache
# %% App state


class _AppState:
    client: InferenceClient
    mongo: MongoStore
    qdrant: QdrantStore
    loader: DocumentLoader
    ingestor: IngestionPipeline
    rag: RAGPipeline


app_state = _AppState()
# 1 000 sessions max; each expires after 1 hour of inactivity

_session_history = TTLCache[str, list[dict[str, str]], float](maxsize=1000, ttl=3600)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialise all clients and pipelines on startup."""
    groq_cfg = GroqConfig()
    hf_cfg = HuggingFaceConfig()
    mongo_cfg = MongoConfig()
    qdrant_cfg = QdrantConfig()

    print(f"[config] LLM (Groq):  {groq_cfg.model}")
    print(f"[config] Embedding:   {hf_cfg.model_name}  (device={hf_cfg.device})")

    app_state.client = InferenceClient.from_config(groq_cfg, hf_cfg)
    app_state.qdrant = QdrantStore.from_config(app_state.client, qdrant_cfg)
    app_state.mongo = MongoStore.from_config(mongo_cfg)
    app_state.loader = DocumentLoader(app_state.client)
    app_state.ingestor = IngestionPipeline(
        loader=app_state.loader,
        mongo=app_state.mongo,
        qdrant=app_state.qdrant,
    )

    app_state.rag = RAGPipeline(
        client=app_state.client,
        qdrant=app_state.qdrant,
        mongo=app_state.mongo,
    )

    yield


# %% FastAPI app

app = FastAPI(
    title="RAG API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this when you add auth
    allow_methods=["*"],
    allow_headers=["*"],
)


# %% Request / Response models


class ChatRequest(BaseModel):
    query: str
    top_k: int = 5
    session_id: str | None = None


class SourceItem(BaseModel):
    chunk_id: str
    doc_id: str
    score: float
    filename: str = ""


class ChatResponse(BaseModel):
    answer: str
    confidence_score: float
    tool_calls_made: int
    sources: list[SourceItem]
    session_id: str


class DocumentItem(BaseModel):
    id: str
    filename: str
    filetype: str
    char_count: int
    filepath: str


class DeleteResponse(BaseModel):
    doc_id: str
    deleted: dict[str, int]


class HealthResponse(BaseModel):
    mongo_ok: bool
    qdrant_ok: bool
    embedding_ok: bool
    details: dict[str, Any]


# %% Routes — health


@app.get("/v1/health", response_model=HealthResponse)
def health() -> HealthResponse:
    mongo_status = app_state.mongo.health()
    qdrant_status = app_state.qdrant.health()

    return HealthResponse(
        mongo_ok=mongo_status.ok,
        qdrant_ok=qdrant_status.ok,
        embedding_ok=qdrant_status.embedding_ok,
        details={
            "mongo": {
                "document_count": mongo_status.document_count,
                "heading_chunk_count": mongo_status.heading_chunk_count,
                "semantic_chunk_count": mongo_status.semantic_chunk_count,
                "errors": mongo_status.errors,
            },
            "qdrant": {
                "document_count": qdrant_status.document_count,
                "collection": qdrant_status.collection_name,
                "errors": qdrant_status.errors,
            },
        },
    )


# %% Routes — chat


@app.post("/v1/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    session_id = request.session_id or str(uuid.uuid4())
    history = _session_history.get(session_id) or []
    _session_history[session_id] = history

    result = app_state.rag.run(request.query, history=history)

    history.append({"role": "user", "content": request.query})
    history.append({"role": "assistant", "content": result["answer"]})

    return ChatResponse(
        answer=result["answer"],
        confidence_score=result["confidence_score"],
        tool_calls_made=result["tool_calls_made"],
        sources=[SourceItem(**s) for s in result["sources"]],
        session_id=session_id,
    )

@app.delete("/v1/chat/{session_id}")
def clear_session(session_id: str) -> dict[str, str]:
    _session_history.pop(session_id, None)
    return {"session_id": session_id, "status": "cleared"}

@app.post("/v1/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """
    Streaming chat endpoint. Yields the answer token by token using
    server-sent events (SSE). Sources and metadata are sent as a
    final 'done' event.
    """

    async def generate() -> AsyncGenerator[str, None]:
        # Run the full RAG pipeline first to resolve tool calls,
        # then stream the final answer token by token via Ollama
        session_id = request.session_id or str(uuid.uuid4())
        history = _session_history.get(session_id) or []
        _session_history[session_id] = history
        result = app_state.rag.run(request.query, history=history)
        history.append({"role": "user", "content": request.query})
        history.append({"role": "assistant", "content": result["answer"]})

        # Stream the answer character by character
        # (swap for ollama stream when tool-call loop is async)
        for char in result["answer"]:
            yield f"data: {char}\n\n"

        # Final metadata event
        import json

        meta = {
            "done": True,
            "confidence_score": result["confidence_score"],
            "tool_calls_made": result["tool_calls_made"],
            "sources": result["sources"],
            "session_id": session_id,
        }
        yield f"data: {json.dumps(meta)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# %% Routes — documents


@app.get("/v1/documents", response_model=list[DocumentItem])
def list_documents() -> list[DocumentItem]:
    docs = app_state.mongo.list_documents()
    return [
        DocumentItem(
            id=d.id,
            filename=d.filename,
            filetype=d.filetype,
            char_count=d.char_count,
            filepath=d.filepath,
        )
        for d in docs
    ]


@app.post("/v1/documents/upload", response_model=DocumentItem)
async def upload_document(file: UploadFile = File(...)) -> DocumentItem:
    """
    Upload and ingest a document. Supported: .pdf, .md, .txt
    """
    filename = file.filename or "upload"
    suffix = os.path.splitext(filename)[-1].lower()

    if suffix not in {".pdf", ".md", ".txt"}:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {suffix}. Supported: .pdf, .md, .txt",
        )

    # Write to a temp file so DocumentLoader can read it from disk
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    # Rename so DocumentLoader picks up the correct filename in metadata
    named_path = os.path.join(tempfile.gettempdir(), filename)
    os.rename(tmp_path, named_path)

    try:
        processed = app_state.ingestor.ingest_file(named_path)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(named_path):
            os.remove(named_path)

    doc = processed.document
    return DocumentItem(
        id=doc.id,
        filename=doc.filename,
        filetype=doc.filetype,
        char_count=doc.char_count,
        filepath=doc.filepath,
    )


@app.delete("/v1/documents/{doc_id}", response_model=DeleteResponse)
def delete_document(doc_id: str) -> DeleteResponse:
    doc = app_state.mongo.get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

    counts = app_state.ingestor.delete_document(doc_id)
    return DeleteResponse(doc_id=doc_id, deleted=counts)


@app.post("/v1/documents/{doc_id}/reingest", response_model=DocumentItem)
async def reingest_document(doc_id: str, file: UploadFile = File(...)) -> DocumentItem:
    """
    Delete an existing document and re-ingest a new version.
    """
    doc = app_state.mongo.get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

    filename = file.filename or doc.filename
    suffix = os.path.splitext(filename)[-1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    named_path = os.path.join(tempfile.gettempdir(), filename)
    os.rename(tmp_path, named_path)

    try:
        processed = app_state.ingestor.reingest_file(named_path, doc_id=doc_id)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(named_path):
            os.remove(named_path)

    d = processed.document
    return DocumentItem(
        id=d.id,
        filename=d.filename,
        filetype=d.filetype,
        char_count=d.char_count,
        filepath=d.filepath,
    )


# %% Entry point

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=True)

