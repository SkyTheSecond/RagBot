# %% Imports
from __future__ import annotations

from typing import TYPE_CHECKING

from inference_client import InferenceClient
from document_loader import DocumentLoader, ProcessedDocument
from mongo_store import MongoStore
from qdrant_store import QdrantStore

if TYPE_CHECKING:
    from config import MongoConfig, QdrantConfig, GroqConfig, HuggingFaceConfig


# %% IngestionPipeline

class IngestionPipeline:
    """
    Orchestrates document ingestion across MongoStore and QdrantStore.

    Responsibilities:
        - Load and chunk documents via DocumentLoader
        - Store heading chunks + full document in MongoDB
        - Store semantic chunks (with embeddings) in QdrantDB
        - Handle delete and reingest for individual documents

    MongoStore receives:  ParsedDocument, HeadingChunk, SemanticChunk
    QdrantStore receives: SemanticChunk (embedded)
    """

    def __init__(
        self,
        loader: DocumentLoader,
        mongo: MongoStore,
        qdrant: QdrantStore,
    ) -> None:
        self._loader = loader
        self._mongo  = mongo
        self._qdrant = qdrant

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def ingest_directory(self, path: str) -> dict[str, int]:
        """
        Load and ingest all supported documents from a directory.

        Returns:
            Summary dict with counts of ingested, skipped, and failed documents.
        """
        results = self._loader.load_directory(path)

        ingested = 0
        skipped  = 0
        failed   = 0

        print(f"\nIngesting {len(results)} document(s) into stores...")
        print("─" * 50)

        for processed in results:
            name = processed.document.filename
            try:
                already = self._mongo.get_document(processed.document.id)
                if already is not None:
                    print(f"  ⏭  {name} — already ingested, skipping")
                    skipped += 1
                    continue

                self._ingest_one(processed)
                print(
                    f"  ✅ {name} — "
                    f"{len(processed.heading_chunks)} sections, "
                    f"{len(processed.semantic_chunks)} chunks"
                )
                ingested += 1

            except Exception as e:
                print(f"  ❌ {name} — FAILED: {e}")
                failed += 1

        print("─" * 50)
        print(
            f"Done. {ingested} ingested, {skipped} skipped, {failed} failed.\n"
        )

        return {"ingested": ingested, "skipped": skipped, "failed": failed}

    def ingest_file(self, filepath: str) -> ProcessedDocument:
        """
        Load and ingest a single file.

        Returns:
            The ProcessedDocument that was ingested.

        Raises:
            ValueError: If the file is a scanned PDF or unsupported type.
            RuntimeError: If ingestion into either store fails.
        """
        processed = self._loader.load_file(filepath)
        name = processed.document.filename

        existing = self._mongo.get_document(processed.document.id)
        if existing is not None:
            print(f"⏭  {name} already ingested — skipping")
            return processed

        try:
            self._ingest_one(processed)
            print(
                f"✅ {name} ingested — "
                f"{len(processed.heading_chunks)} sections, "
                f"{len(processed.semantic_chunks)} chunks"
            )
        except Exception as e:
            raise RuntimeError(f"Ingestion failed for {name}: {e}") from e

        return processed

    def _ingest_one(self, processed: ProcessedDocument) -> None:
        """Write a ProcessedDocument to both stores."""
        # Mongo: full document + heading chunks + semantic chunks
        self._mongo.store_all(
            document=processed.document,
            heading_chunks=processed.heading_chunks,
            semantic_chunks=processed.semantic_chunks,
        )
        # Qdrant: semantic chunks only (embedded internally)
        self._qdrant.store(processed.semantic_chunks)

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_document(self, doc_id: str) -> dict[str, int]:
        """
        Delete a document and all its chunks from both stores.

        Args:
            doc_id: The document ID (ParsedDocument.id).

        Returns:
            Combined deletion counts from both stores.
        """
        mongo_counts  = self._mongo.delete_document(doc_id)
        qdrant_deleted = self._qdrant.delete_document(doc_id)

        return {
            "mongo_documents":       mongo_counts["documents"],
            "mongo_heading_chunks":  mongo_counts["heading_chunks"],
            "mongo_semantic_chunks": mongo_counts["semantic_chunks"],
            "qdrant_chunks":         qdrant_deleted,
        }

    def delete_by_filename(self, filename: str) -> dict[str, int] | None:
        """
        Convenience method — delete a document by filename instead of ID.

        Returns:
            Deletion counts, or None if the document was not found.
        """
        doc = self._mongo.get_document_by_filename(filename)
        if doc is None:
            print(f"⚠  No document found with filename: {filename}")
            return None
        return self.delete_document(doc.id)

    # ------------------------------------------------------------------
    # Reingest
    # ------------------------------------------------------------------

    def reingest_file(self, filepath: str, doc_id: str | None = None) -> ProcessedDocument:
        """
        Delete an existing document by filename and re-ingest it from disk.
        Useful when a document has been updated on disk.
        Pass doc_id directly (preferred) to avoid a secondary filename lookup.

        Returns:
            The freshly ingested ProcessedDocument.
        """
        import os
        filename = os.path.basename(filepath)

        if doc_id is not None:
            existing_id = doc_id
            print(f"🗑  Deleting existing document (id={doc_id}): {filename}")
        else:
            existing = self._mongo.get_document_by_filename(filename)
            existing_id = existing.id if existing is not None else None
            if existing_id is None:
                print(f"⚠  {filename} not found in store — ingesting fresh")

        if existing_id is not None:
            counts = self.delete_document(existing_id)
            print(
                f"   Removed — "
                f"mongo: {counts['mongo_semantic_chunks']} chunks, "
                f"qdrant: {counts['qdrant_chunks']} chunks"
            )
        return self.ingest_file(filepath)

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health(self) -> dict[str, bool]:
        """Quick combined health check across both stores."""
        mongo_status  = self._mongo.health()
        qdrant_status = self._qdrant.health()

        print(f"MongoDB:   {'✅' if mongo_status.ok  else '❌'}")
        print(f"Qdrant:  {'✅' if qdrant_status.ok else '❌'}")

        if mongo_status.errors:
            for e in mongo_status.errors:
                print(f"  • Mongo error: {e}")
        if qdrant_status.errors:
            for e in qdrant_status.errors:
                print(f"  • Qdrant error: {e}")

        return {
            "mongo_ok":  mongo_status.ok,
            "qdrant_ok": qdrant_status.ok,
        }


# %% Heartbeat

if __name__ == "__main__":
    from config import MongoConfig, QdrantConfig, GroqConfig, HuggingFaceConfig

    client  = InferenceClient.from_config(GroqConfig(), HuggingFaceConfig())
    mongo   = MongoStore.from_config(MongoConfig())
    qdrant  = QdrantStore.from_config(client, QdrantConfig())
    loader  = DocumentLoader(client)

    pipeline = IngestionPipeline(loader=loader, mongo=mongo, qdrant=qdrant)

    print("── Health ──")
    pipeline.health()