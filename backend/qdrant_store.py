# %% Imports
import uuid
from typing import Any, Sequence

from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    PayloadSchemaType,
    VectorParams,
)

from config import QdrantConfig
from document_loader import SemanticChunk
from inference_client import InferenceClient


def _str_to_uuid(s: str) -> str:
    """Deterministically convert a string chunk ID to a UUID for Qdrant."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, s))


class QueryResult(BaseModel):
    chunk_id: str
    content: str
    score: float  # cosine similarity in [-1, 1]; higher = more similar
    doc_id: str
    section_id: str
    position: int
    prev_id: str | None
    next_id: str | None
    filename: str = ""
    metadata: dict[str, Any]


class HealthStatus(BaseModel):
    qdrant_ok: bool
    embedding_ok: bool
    collection_name: str
    document_count: int
    errors: list[str]

    @property
    def ok(self) -> bool:
        return self.qdrant_ok and self.embedding_ok


class QdrantStore:
    """
    Wraps a Qdrant collection with typed store, query, and health methods.

    Args:
        embedder:         InferenceClient instance (used for embedding).
        collection_name:  Name of the Qdrant collection to use or create.
        host:             Qdrant host.
        port:             Qdrant port (default 6333).
    """

    def __init__(
            self,
            embedder: InferenceClient,
            collection_name: str = "rag_chunks",
            host: str = "localhost",
            port: int = 6333,
            api_key: str | None = None,
        ) -> None:
        self._embedder = embedder
        self._collection_name = collection_name

        # Check if connecting to Qdrant Cloud or an explicitly secure instance
        if host and ("cloud.qdrant.io" in host or host.startswith("https://")):
            # Clean up the host string just in case protocols were pre-attached
            clean_host = host.replace("https://", "").replace("http://", "")
            # Force the secure https:// protocol on the cloud cluster
            url = f"https://{clean_host}:{port}"
            self._client = QdrantClient(url=url, api_key=api_key)
        else:
            # Fallback for standard local development deployments (e.g., localhost:6333)
            self._client = QdrantClient(host=host, port=port, api_key=api_key)

        self._ensure_collection()


    def _ensure_collection(self) -> None:
        """Create the collection if it doesn't already exist."""
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection_name not in existing:
            # Probe once to learn the embedding dimension.
            dim = len(self._embedder.embed("probe"))
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
        # doc_id is filtered on in delete_document (and could be filtered on
        # in query); Qdrant requires an explicit payload index for that.
        # create_payload_index is idempotent, so it's fine to call this
        # unconditionally on every startup rather than only on creation.

        self._client.create_payload_index(
            collection_name=self._collection_name,
            field_name="doc_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )


    @classmethod
    def from_config(
        cls, embedder: "InferenceClient", config: "QdrantConfig"
    ) -> "QdrantStore":
        return cls(
            embedder=embedder,
            collection_name=config.collection_name,
            host=config.host,
            port=config.port,
            api_key=config.api_key,
        )

    def store(self, chunks: list[SemanticChunk]) -> None:
        if not chunks:
            return

        # Build UUID → chunk map; UUIDs are the actual Qdrant point IDs.
        id_map = {_str_to_uuid(c.id): c for c in chunks}
        existing = {
            str(r.id)
            for r in self._client.retrieve(
                collection_name=self._collection_name,
                ids=list(id_map.keys()),
            )
        }
        new_chunks = [c for uid, c in id_map.items() if uid not in existing]

        if not new_chunks:
            return

        texts = [c.content for c in new_chunks]
        embeddings = self._embedder.embed_batch(texts)

        self._client.upsert(
            collection_name=self._collection_name,
            points=[
                PointStruct(
                    id=_str_to_uuid(c.id),
                    vector=list(emb),
                    payload=_chunk_payload(c),
                )
                for c, emb in zip(new_chunks, embeddings)
            ],
        )

    def delete_document(self, doc_id: str) -> int:
        before = self._client.count(collection_name=self._collection_name).count
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            ),
        )
        after = self._client.count(collection_name=self._collection_name).count
        return before - after

    def query(self, text: str, k: int = 5) -> list[QueryResult]:
        """
        Embed a query string and return the top-k most similar chunks.
        """
        embedding: Sequence[float] = self._embedder.embed(text)

        hits = self._client.query_points(
            collection_name=self._collection_name,
            query=list(embedding),
            limit=k,
            with_payload=True,
        ).points

        results: list[QueryResult] = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(
                QueryResult(
                    chunk_id=str(payload.get("chunk_id", "")),
                    content=str(payload.get("content", "")),
                    score=hit.score,
                    doc_id=str(payload.get("doc_id", "")),
                    section_id=str(payload.get("section_id", "")),
                    position=int(payload.get("position", 0)),
                    prev_id=payload.get("prev_id") or None,
                    next_id=payload.get("next_id") or None,
                    filename=str(payload.get("source", "")),
                    metadata={
                        key: v
                        for key, v in payload.items()
                        if key
                        not in {
                            "chunk_id",
                            "content",
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

    def health(self) -> HealthStatus:
        errors: list[str] = []
        qdrant_ok = False
        embedding_ok = False
        document_count = 0

        try:
            self._client.get_collections()
            document_count = self._client.count(
                collection_name=self._collection_name
            ).count
            qdrant_ok = True
        except Exception as e:
            errors.append(f"Qdrant: {e}")

        try:
            vec = self._embedder.embed("health check")
            if not vec or len(vec) == 0:
                errors.append("Embedding: returned empty vector")
            else:
                embedding_ok = True
        except Exception as e:
            errors.append(f"Embedding: {e}")

        return HealthStatus(
            qdrant_ok=qdrant_ok,
            embedding_ok=embedding_ok,
            collection_name=self._collection_name,
            document_count=document_count,
            errors=errors,
        )


def _chunk_payload(chunk: SemanticChunk) -> dict[str, Any]:
    # Qdrant has no separate documents field — content lives in the payload.
    return {
        "chunk_id": chunk.id,  # original string ID; point ID is its UUID5
        "content": chunk.content,
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
    from config import GroqConfig, HuggingFaceConfig, QdrantConfig
    from inference_client import InferenceClient

    embedder = InferenceClient.from_config(GroqConfig(), HuggingFaceConfig())
    store = QdrantStore.from_config(embedder, QdrantConfig())
    status = store.health()

    print(f"Qdrant:    {'✅' if status.qdrant_ok    else '❌'}")
    print(f"Embedding: {'✅' if status.embedding_ok else '❌'}")
    print(f"Collection: {status.collection_name}  ({status.document_count} docs)")

    if status.errors:
        print("\nErrors:")
        for err in status.errors:
            print(f"  • {err}")
    else:
        print("\n🎉 All systems go")
