# %% Imports
from typing import Any, Sequence

import chromadb
import numpy as np
from chromadb.api import ClientAPI
from chromadb.types import Metadata  # their own alias
from pydantic import BaseModel
from tqdm import tqdm

from config import ChromaConfig
from document_loader import SemanticChunk
from inference_client import InferenceClient

class QueryResult(BaseModel):
    chunk_id: str
    content: str
    score: float
    doc_id: str
    section_id: str
    position: int
    prev_id: str | None
    next_id: str | None
    filename: str = ""
    metadata: dict[str, Any]


class HealthStatus(BaseModel):
    chroma_ok: bool
    embedding_ok: bool
    collection_name: str
    document_count: int
    errors: list[str]

    @property
    def ok(self) -> bool:
        return self.chroma_ok and self.embedding_ok


class ChromaStore:
    """
    Wraps a ChromaDB HTTP collection with typed store, query, and health methods.

    Args:
        ollama_client:    OllamaClient instance (used for embedding).
        collection_name:  Name of the Chroma collection to use or create.
        host:             ChromaDB HTTP host.
        port:             ChromaDB HTTP port.
    """

    def __init__(
        self,
        embedder: InferenceClient,
        collection_name: str = "rag_chunks",
        host: str = "localhost",
        port: int = 8000,
    ) -> None:
        self._embedder = embedder
        self._collection_name = collection_name
        self._client: ClientAPI = chromadb.HttpClient(host=host, port=port)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # cosine similarity
        )

    @classmethod
    def from_config(
        cls, embedder: "InferenceClient", config: "ChromaConfig"
    ) -> "ChromaStore":
        return cls(
            embedder=embedder,
            collection_name=config.collection_name,
            host=config.host,
            port=config.port,
        )

    def store(self, chunks: list[SemanticChunk]) -> None:
        if not chunks:
            return

        existing = set(self._collection.get(ids=[c.id for c in chunks])["ids"])
        new_chunks = [c for c in chunks if c.id not in existing]

        if not new_chunks:
            return

        texts = [c.content for c in new_chunks]
        embeddings = self._embedder.embed_batch(texts)

        self._collection.add(
            ids=[c.id for c in new_chunks],
            embeddings=embeddings,  # type: ignore[arg-type]
            documents=texts,
            metadatas=[_chunk_metadata(c) for c in new_chunks],
        )

    def delete_document(self, doc_id: str) -> int:
        before = self._collection.count()
        self._collection.delete(where={"doc_id": doc_id})
        after = self._collection.count()
        return before - after

    def query(self, text: str, k: int = 5) -> list[QueryResult]:
        """
        Embed a query string and return the top-k most similar chunks.
        """
        embedding: Sequence[float] = self._embedder.embed(text)

        raw = self._collection.query(
            query_embeddings=[embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        results: list[QueryResult] = []

        ids = (raw.get("ids") or [[]])[0]
        documents = (raw.get("documents") or [[]])[0]
        metadatas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]

        chunk_id: str
        doc: str
        meta: Metadata
        distance: float

        for chunk_id, doc, meta, distance in zip(ids, documents, metadatas, distances):
            score = 1.0 - (float(distance) / 2.0)
            meta = meta or {}

            results.append(
                QueryResult(
                    chunk_id=chunk_id,
                    content=doc,
                    score=score,
                    doc_id=str(meta.get("doc_id", "")),
                    section_id=str(meta.get("section_id", "")),
                    position=int(meta.get("position", 0)),  # type: ignore[arg-type]
                    prev_id=meta.get("prev_id") or None,  # type: ignore[arg-type]
                    next_id=meta.get("next_id") or None,  # type: ignore[arg-type]
                    filename=str(meta.get("source", "")),
                    metadata={
                        k: v
                        for k, v in meta.items()
                        if k
                        not in {
                            "doc_id",
                            "section_id",
                            "position",
                            "prev_id",
                            "next_id",
                            "source",
                        }
                    },
                )
            )

        return results

    @staticmethod
    def safe_int(v: Any, default: int = 0) -> int:
        try:
            return int(v)
        except (TypeError, ValueError):
            return default

    def health(self) -> HealthStatus:
        """
        Check whether Chroma and the embedding model are reachable.

        Returns:
            HealthStatus with per-component flags and any error messages.
        """
        errors: list[str] = []
        chroma_ok = False
        embedding_ok = False
        document_count = 0

        # Check Chroma
        try:
            self._client.heartbeat()
            document_count = self._collection.count()
            chroma_ok = True
        except Exception as e:
            errors.append(f"Chroma: {e}")

        # Check embedding model via a minimal embed call
        try:
            vec = self._embedder.embed("health check")
            if not vec or len(vec) == 0:
                errors.append("Embedding: returned empty vector")
            else:
                embedding_ok = True
        except Exception as e:
            errors.append(f"Embedding: {e}")

        return HealthStatus(
            chroma_ok=chroma_ok,
            embedding_ok=embedding_ok,
            collection_name=self._collection_name,
            document_count=document_count,
            errors=errors,
        )


def _chunk_metadata(chunk: SemanticChunk) -> dict[str, Any]:
    return {
        "doc_id": chunk.doc_id,
        "section_id": chunk.section_id,
        "position": chunk.position,
        "prev_id": chunk.prev_id or "",
        "next_id": chunk.next_id or "",
        **{
            k: v
            for k, v in chunk.metadata.items()
            if isinstance(v, (str, int, float, bool))
        },
    }


# %% Heartbeat

if __name__ == "__main__":
    from config import ChromaConfig, GroqConfig, HuggingFaceConfig
    from inference_client import InferenceClient
    embedder = InferenceClient.from_config(GroqConfig(), HuggingFaceConfig())

    store = ChromaStore.from_config(embedder, ChromaConfig())

    status = store.health()

    print(f"Chroma:    {'✅' if status.chroma_ok    else '❌'}")
    print(f"Embedding: {'✅' if status.embedding_ok else '❌'}")
    print(f"Collection: {status.collection_name}  ({status.document_count} docs)")

    if status.errors:
        print("\nErrors:")
        for err in status.errors:
            print(f"  • {err}")
    else:
        print("\n🎉 All systems go")

