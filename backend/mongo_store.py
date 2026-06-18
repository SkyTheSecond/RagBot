# %% Imports
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config import MongoConfig

from typing import Any

from pydantic import BaseModel
from pymongo import ASCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from tqdm import tqdm

from document_loader import HeadingChunk, ParsedDocument, SemanticChunk

# %% Models


class HealthStatus(BaseModel):
    mongo_ok: bool
    collection_name: str
    document_count: int
    heading_chunk_count: int
    semantic_chunk_count: int
    errors: list[str]

    @property
    def ok(self) -> bool:
        return self.mongo_ok


# %% MongoStore


class MongoStore:
    """
    Wraps a MongoDB database with three collections:

        documents      → ParsedDocument (full extracted text + metadata)
        heading_chunks → HeadingChunk   (document sections by heading)
        semantic_chunks → SemanticChunk  (chunks that live in Chroma too)

    All writes are idempotent — re-ingesting the same document is safe.
    """

    def __init__(
        self,
        uri: str = "mongodb://localhost:27017",
        db_name: str = "rag",
    ) -> None:
        self._client: MongoClient[dict[str, Any]] = MongoClient(uri)
        self._db: Database[dict[str, Any]] = self._client[db_name]

        self._documents: Collection[dict[str, Any]] = self._db["documents"]
        self._heading_chunks: Collection[dict[str, Any]] = self._db["heading_chunks"]
        self._semantic_chunks: Collection[dict[str, Any]] = self._db["semantic_chunks"]

        self._ensure_indexes()

    @classmethod
    def from_config(cls, config: "MongoConfig") -> "MongoStore":
        return cls(uri=config.uri, db_name=config.db_name)

    # ------------------------------------------------------------------
    # Indexes
    # ------------------------------------------------------------------

    def _ensure_indexes(self) -> None:
        # documents
        self._documents.create_index("id", unique=True)
        self._documents.create_index("filename")

        # heading_chunks
        self._heading_chunks.create_index("id", unique=True)
        self._heading_chunks.create_index("doc_id")

        # semantic_chunks
        self._semantic_chunks.create_index("id", unique=True)
        self._semantic_chunks.create_index("doc_id")
        self._semantic_chunks.create_index("section_id")
        self._semantic_chunks.create_index(
            [("doc_id", ASCENDING), ("position", ASCENDING)]
        )

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    def store_document(self, document: ParsedDocument) -> None:
        """Upsert a ParsedDocument. Safe to call multiple times."""
        self._documents.update_one(
            {"id": document.id},
            {"$set": document.model_dump()},
            upsert=True,
        )

    def store_heading_chunks(self, chunks: list[HeadingChunk]) -> None:
        """Upsert a list of HeadingChunks."""
        for chunk in tqdm(chunks, desc="[mongo] heading chunks", unit="chunk"):
            self._heading_chunks.update_one(
                {"id": chunk.id},
                {"$set": chunk.model_dump()},
                upsert=True,
            )

    def store_semantic_chunks(self, chunks: list[SemanticChunk]) -> None:
        """Upsert a list of SemanticChunks."""
        for chunk in tqdm(chunks, desc="[mongo] semantic chunks", unit="chunk"):
            self._semantic_chunks.update_one(
                {"id": chunk.id},
                {"$set": chunk.model_dump()},
                upsert=True,
            )

    def store_all(
        self,
        document: ParsedDocument,
        heading_chunks: list[HeadingChunk],
        semantic_chunks: list[SemanticChunk],
    ) -> None:
        """Convenience method — store everything for a ProcessedDocument at once."""
        self.store_document(document)
        self.store_heading_chunks(heading_chunks)
        self.store_semantic_chunks(semantic_chunks)

    # ------------------------------------------------------------------
    # Fetch — documents
    # ------------------------------------------------------------------

    def get_document(self, doc_id: str) -> ParsedDocument | None:
        """Fetch a full document by ID."""
        raw = self._documents.find_one({"id": doc_id}, {"_id": 0})
        return ParsedDocument(**raw) if raw else None

    def get_document_by_filename(self, filename: str) -> ParsedDocument | None:
        """Fetch a document by its filename."""
        raw = self._documents.find_one({"filename": filename}, {"_id": 0})
        return ParsedDocument(**raw) if raw else None

    def list_documents(self) -> list[ParsedDocument]:
        """Return all documents (without full content — just metadata)."""
        raws = self._documents.find(
            {},
            {"_id": 0, "content": 0},  # exclude heavy content field
        )
        return [ParsedDocument(**r) for r in raws]

    # ------------------------------------------------------------------
    # Fetch — heading chunks
    # ------------------------------------------------------------------

    def get_heading_chunk(self, section_id: str) -> HeadingChunk | None:
        """Fetch a single heading chunk by ID."""
        raw = self._heading_chunks.find_one({"id": section_id}, {"_id": 0})
        return HeadingChunk(**raw) if raw else None

    def get_heading_chunks_for_document(self, doc_id: str) -> list[HeadingChunk]:
        """Return all heading chunks for a document, ordered by start_char."""
        raws = self._heading_chunks.find(
            {"doc_id": doc_id},
            {"_id": 0},
            sort=[("start_char", ASCENDING)],
        )
        return [HeadingChunk(**r) for r in raws]

    # ------------------------------------------------------------------
    # Fetch — semantic chunks
    # ------------------------------------------------------------------

    def get_semantic_chunk(self, chunk_id: str) -> SemanticChunk | None:
        """Fetch a single semantic chunk by ID."""
        raw = self._semantic_chunks.find_one({"id": chunk_id}, {"_id": 0})
        return SemanticChunk(**raw) if raw else None

    def get_semantic_chunks_for_document(self, doc_id: str) -> list[SemanticChunk]:
        """Return all semantic chunks for a document, ordered by position."""
        raws = self._semantic_chunks.find(
            {"doc_id": doc_id},
            {"_id": 0},
            sort=[("position", ASCENDING)],
        )
        return [SemanticChunk(**r) for r in raws]

    def get_semantic_chunks_for_section(self, section_id: str) -> list[SemanticChunk]:
        """Return all semantic chunks belonging to a heading section."""
        raws = self._semantic_chunks.find(
            {"section_id": section_id},
            {"_id": 0},
            sort=[("position", ASCENDING)],
        )
        return [SemanticChunk(**r) for r in raws]

    def get_surrounding_chunks(self, chunk_id: str, n: int = 2) -> list[SemanticChunk]:
        """
        Walk the prev/next linked list n steps in each direction from chunk_id.
        Returns chunks ordered by position, including the anchor chunk itself.
        """
        anchor = self.get_semantic_chunk(chunk_id)
        if anchor is None:
            return []

        collected: dict[str, SemanticChunk] = {anchor.id: anchor}

        # Walk backwards
        current = anchor
        for _ in range(n):
            if current.prev_id is None:
                break
            prev = self.get_semantic_chunk(current.prev_id)
            if prev is None:
                break
            collected[prev.id] = prev
            current = prev

        # Walk forwards
        current = anchor
        for _ in range(n):
            if current.next_id is None:
                break
            nxt = self.get_semantic_chunk(current.next_id)
            if nxt is None:
                break
            collected[nxt.id] = nxt
            current = nxt

        return sorted(collected.values(), key=lambda c: c.position)

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_document(self, doc_id: str) -> dict[str, int]:
        """
        Delete a document and all its heading and semantic chunks.

        Returns:
            Dict with counts of each deleted collection entry.
        """
        docs_deleted = self._documents.delete_one({"id": doc_id}).deleted_count
        headings_deleted = self._heading_chunks.delete_many(
            {"doc_id": doc_id}
        ).deleted_count
        chunks_deleted = self._semantic_chunks.delete_many(
            {"doc_id": doc_id}
        ).deleted_count

        return {
            "documents": docs_deleted,
            "heading_chunks": headings_deleted,
            "semantic_chunks": chunks_deleted,
        }

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health(self) -> HealthStatus:
        """Check whether MongoDB is reachable and return collection counts."""
        errors: list[str] = []
        mongo_ok = False
        doc_count = heading_count = semantic_count = 0

        try:
            self._client.admin.command("ping")
            doc_count = self._documents.count_documents({})
            heading_count = self._heading_chunks.count_documents({})
            semantic_count = self._semantic_chunks.count_documents({})
            mongo_ok = True
        except Exception as e:
            errors.append(f"MongoDB: {e}")

        return HealthStatus(
            mongo_ok=mongo_ok,
            collection_name=self._db.name,
            document_count=doc_count,
            heading_chunk_count=heading_count,
            semantic_chunk_count=semantic_count,
            errors=errors,
        )


# %% Heartbeat

if __name__ == "__main__":
    from config import MongoConfig

    store = MongoStore.from_config(MongoConfig())
    status = store.health()

    print(f"MongoDB:         {'✅' if status.mongo_ok else '❌'}")
    print(f"Database:         {status.collection_name}")
    print(f"Documents:        {status.document_count}")
    print(f"Heading chunks:   {status.heading_chunk_count}")
    print(f"Semantic chunks:  {status.semantic_chunk_count}")

    if status.errors:
        print("\nErrors:")
        for err in status.errors:
            print(f"  • {err}")
    else:
        print("\n🎉 All systems go")
