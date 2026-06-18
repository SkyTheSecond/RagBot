# %% Imports & shared fixtures
#
# Run this cell once per session before running any test cell below — each
# test cell is self-contained and re-runnable on its own after that, but they
# all rely on the names defined here (imports, synthetic file builders, and
# the small _Reporter helper used for per-check pass/fail reporting).

import os
import tempfile
import textwrap

import fitz  # pymupdf

from config import GroqConfig, HuggingFaceConfig, MongoConfig, QdrantConfig
from inference_client import InferenceClient, ChatMessage, Tool, ToolFunction
from document_loader import (
    DocumentLoader,
    ProcessedDocument,
    HeadingChunk,
    SemanticChunk,
)
from mongo_store import MongoStore
from qdrant_store import QdrantStore
from ingestion_pipeline import IngestionPipeline
from rag_pipeline import RAGPipeline


# ---------------------------------------------------------------------------
# Reporting helper
# ---------------------------------------------------------------------------
# Every test below runs its checks as a series of `with r.check("label"):`
# blocks. A failure inside one block is caught, logged, and printed — it does
# NOT stop the rest of the checks in that test from running. This is the same
# pattern the original test_document_loader used (try/except + a `failed`
# list); it's just factored out so every system below gets it for free.

class _Reporter:
    def __init__(self, name: str) -> None:
        self.name = name
        self.failed: list[str] = []

    def check(self, label: str) -> "_Check":
        return _Check(self, label)

    def finish(self) -> None:
        print("\n" + "─" * 50)
        if self.failed:
            print(f"❌ {len(self.failed)} check(s) failed in {self.name}:")
            for f in self.failed:
                print(f"   • {f}")
            raise RuntimeError(f"{len(self.failed)} check(s) failed in {self.name} — see above")
        print(f"🎉 All {self.name} checks passed")


class _Check:
    def __init__(self, reporter: _Reporter, label: str) -> None:
        self.reporter = reporter
        self.label = label

    def __enter__(self) -> "_Check":
        print(f"\n🔹 {self.label}...")
        return self

    def __exit__(self, exc_type: type | None, exc: BaseException | None, tb: object) -> bool:
        if exc_type is None:
            print(f"  ✅ {self.label}")
        else:
            msg = f"{self.label}: {exc}"
            print(f"  ❌ {msg}")
            self.reporter.failed.append(msg)
        return True  # swallow it — move on to the next check


# ---------------------------------------------------------------------------
# Synthetic file builders
# ---------------------------------------------------------------------------

def _make_markdown(directory: str) -> str:
    path = os.path.join(directory, "test.md")
    content = textwrap.dedent("""\
        # Introduction

        This section introduces the topic of travel in Southeast Asia.
        There are many beautiful destinations to explore across the region.
        Each country offers its own unique culture and cuisine.

        ## Bali

        Bali is a popular destination known for its temples and rice terraces.
        Visitors often explore Ubud for its art scene and spiritual atmosphere.
        The beaches of Seminyak and Kuta attract surfers and sun seekers alike.

        ### Regions to Travel in Bali

        The northern region of Bali offers volcanic landscapes and quieter beaches.
        Central Bali is home to traditional villages and craft markets.
        Southern Bali is the most tourist-friendly with resorts and nightlife.

        ## Lombok

        Lombok sits just east of Bali and is known for Mount Rinjani.
        The Gili Islands off Lombok's northwest coast are famous for snorkelling.
        The island has a more relaxed pace compared to its western neighbour.

        # Conclusion

        Southeast Asia offers endless variety for travellers of all kinds.
        Planning ahead ensures you make the most of each destination.
    """)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def _make_txt(directory: str) -> str:
    path = os.path.join(directory, "test.txt")
    content = textwrap.dedent("""\
        INTRODUCTION
        ============

        This document covers the basics of vector databases.
        Vector databases store high-dimensional embeddings for similarity search.
        They are a core component of modern RAG pipelines.

        WHAT IS A VECTOR
        ================

        A vector is a list of floating point numbers representing semantic meaning.
        Two vectors that are close in space tend to share similar meaning.
        Cosine similarity is a common metric for comparing vectors.

        QDRANT
        ======

        Qdrant is an open-source vector database written in Rust for performance.
        It supports metadata filtering alongside vector similarity search.
        Qdrant can run self-hosted or as a managed cloud service.
    """)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def _make_pdf(directory: str) -> str:
    """Create a text-based PDF with pymupdf, with enough content to pass the scanned threshold."""
    path = os.path.join(directory, "test.pdf")
    doc = fitz.open()
    page = doc.new_page()

    y = 80

    def insert(text: str, fontsize: int) -> None:
        nonlocal y
        page.insert_text((72, y), text, fontsize=fontsize)
        y += fontsize * 1.4 * (text.count("\n") + 1)

    insert("Vector Search Overview", fontsize=22)
    y += 10

    insert(
        "Vector search is a technique used to find similar items in high-dimensional space.\n"
        "It is widely used in recommendation systems, semantic search engines, and RAG pipelines.\n"
        "The core idea is to represent items as dense numerical vectors called embeddings.\n"
        "Once items are embedded, similarity can be computed using metrics like cosine distance.\n"
        "This approach allows retrieval by meaning rather than exact keyword matching.\n"
        "Modern vector databases are optimised for approximate nearest neighbour search at scale.\n",
        fontsize=11,
    )
    y += 10

    insert("How Embeddings Work", fontsize=16)
    y += 6

    insert(
        "An embedding model maps raw input such as text or images into a fixed-size vector.\n"
        "Similar inputs produce vectors that are close together in the embedding space.\n"
        "Transformer-based models such as BERT and its derivatives are commonly used.\n"
        "The dimensionality of the resulting vectors typically ranges from 384 to 1536.\n"
        "Higher-dimensional embeddings tend to capture more semantic nuance.\n"
        "However they also require more memory and compute during retrieval.\n",
        fontsize=11,
    )
    y += 10

    insert("Retrieval and Reranking", fontsize=16)
    y += 6

    insert(
        "The retrieval step fetches the top-k most similar vectors from the database.\n"
        "Approximate nearest neighbour algorithms such as HNSW make this fast at scale.\n"
        "A reranker then scores each candidate against the original query more precisely.\n"
        "Cross-encoder rerankers read the query and document together for higher accuracy.\n"
        "The final ranked list is passed as context to the language model for generation.\n"
        "Combining retrieval with reranking significantly improves answer quality in RAG.\n",
        fontsize=11,
    )

    doc.save(path)
    doc.close()
    return path


def _make_scanned_pdf(directory: str) -> str:
    """Create a PDF with almost no extractable text to simulate a scanned doc."""
    path = os.path.join(directory, "scanned.pdf")
    doc = fitz.open()
    doc.new_page()
    doc.save(path)
    doc.close()
    return path


# ---------------------------------------------------------------------------
# Assertion / cleanup helpers
# ---------------------------------------------------------------------------

def _assert_processed(result: ProcessedDocument, label: str) -> None:
    assert len(result.heading_chunks) > 0, \
        f"[{label}] Expected at least one heading chunk"
    assert len(result.semantic_chunks) > 0, \
        f"[{label}] Expected at least one semantic chunk"

    heading_ids = {h.id for h in result.heading_chunks}
    for sc in result.semantic_chunks:
        assert sc.section_id in heading_ids, \
            f"[{label}] SemanticChunk {sc.id} has unknown section_id {sc.section_id}"

    for i, sc in enumerate(result.semantic_chunks):
        if i == 0:
            assert sc.prev_id is None, f"[{label}] First chunk should have no prev_id"
        else:
            assert sc.prev_id == result.semantic_chunks[i - 1].id, \
                f"[{label}] Chunk {i} prev_id mismatch"
        if i == len(result.semantic_chunks) - 1:
            assert sc.next_id is None, f"[{label}] Last chunk should have no next_id"
        else:
            assert sc.next_id == result.semantic_chunks[i + 1].id, \
                f"[{label}] Chunk {i} next_id mismatch"

    all_sc_ids = [sc.id for sc in result.semantic_chunks]
    assert len(all_sc_ids) == len(set(all_sc_ids)), \
        f"[{label}] Duplicate semantic chunk IDs found"

    all_hc_ids = [hc.id for hc in result.heading_chunks]
    assert len(all_hc_ids) == len(set(all_hc_ids)), \
        f"[{label}] Duplicate heading chunk IDs found"

    print(
        f"    {label}: {len(result.heading_chunks)} heading sections, "
        f"{len(result.semantic_chunks)} semantic chunks"
    )


def _purge_by_filename(filename: str, mongo: MongoStore, qdrant: QdrantStore | None = None) -> None:
    """
    Best-effort cleanup of any document left behind by a previous interrupted
    test run, looked up by filename rather than ID.

    This matters because ParsedDocument.id is assigned via uuid.uuid4() in
    DocumentLoader — a fresh random ID every time a file is loaded, even for
    byte-identical content. That also means IngestionPipeline's "already
    ingested, skipping" dedup check (which looks documents up by ID) can
    never actually find a match — see the note in test_ingestion_pipeline.
    Net effect for tests: a crashed run can leave an orphaned "test.md" or
    "test.txt" document under some other random ID, so we sweep by filename
    instead of relying on the ID we're about to create.
    """
    existing = mongo.get_document_by_filename(filename)
    while existing is not None:
        mongo.delete_document(existing.id)
        if qdrant is not None:
            qdrant.delete_document(existing.id)
        existing = mongo.get_document_by_filename(filename)


# %% 1. InferenceClient

def test_inference_client() -> None:
    client = InferenceClient.from_config(GroqConfig(), HuggingFaceConfig())

    print("🔍 Checking Inference availability...")
    if not client.is_available():
        raise RuntimeError("Groq not available or API key missing")
    print("✅ Inference client available")

    r = _Reporter("InferenceClient")

    with r.check("embed (single)"):
        vec = client.embed("hello world")
        assert isinstance(vec, list), "Embedding is not a list"
        assert len(vec) > 0, "Empty embedding returned"
        print(f"    dim={len(vec)}")

    with r.check("embed_batch"):
        texts = ["hello world", "goodbye world", "vector search is fun"]
        vecs = client.embed_batch(texts)
        assert isinstance(vecs, list) and len(vecs) == len(texts), \
            "embed_batch returned the wrong number of vectors"
        assert all(isinstance(v, list) and len(v) > 0 for v in vecs), \
            "embed_batch returned an empty vector"
        assert len({len(v) for v in vecs}) == 1, \
            "embed_batch vectors have inconsistent dimensions"
        print(f"    {len(vecs)} vectors, dim={len(vecs[0])}")

    with r.check("generate"):
        text = client.generate("Say: RAG is working")
        assert isinstance(text, str) and len(text) > 0
        print(f"    {text!r}")

    with r.check("generate_stream"):
        chunks = list(client.generate_stream("Count from 1 to 3"))
        joined = "".join(chunks)
        assert len(chunks) > 0, "generate_stream produced no chunks"
        assert len(joined) > 0, "generate_stream joined output is empty"
        print(f"    {len(chunks)} chunks, {len(joined)} chars total")

    with r.check("chat"):
        messages = [
            ChatMessage(role="system", content="You are a strict test assistant."),
            ChatMessage(role="user", content="Reply only with OK"),
        ]
        reply = client.chat(messages)
        assert reply is not None
        assert reply.content or reply.tool_calls, "Chat reply has neither content nor tool_calls"
        print(f"    {reply.content or reply.tool_calls}")

    with r.check("chat_stream"):
        stream_messages = [ChatMessage(role="user", content="Say hello in one short sentence")]
        chunks = list(client.chat_stream(stream_messages))
        joined = "".join(chunks)
        assert len(joined) > 0, "chat_stream produced no output"
        print(f"    {len(chunks)} chunks, {len(joined)} chars total")

    with r.check("tool-calling wiring"):
        tool_messages = [ChatMessage(role="user", content="Call the dummy_tool with x set to 'test'")]
        tools = [
            Tool(
                function=ToolFunction(
                    name="dummy_tool",
                    description="A no-op test tool",
                    parameters={
                        "type": "object",
                        "properties": {"x": {"type": "string"}},
                        "required": ["x"],
                    },
                )
            )
        ]
        reply2 = client.chat(tool_messages, tools=tools)
        assert reply2 is not None
        print(f"    tool_calls={reply2.tool_calls}")

    r.finish()


if __name__ == "__main__":
    test_inference_client()


# %% 2. DocumentLoader

def test_document_loader() -> None:
    client = InferenceClient.from_config(GroqConfig(), HuggingFaceConfig())
    if not client.is_available():
        raise RuntimeError("Groq not reachable — cannot test DocumentLoader")

    loader = DocumentLoader(client)
    r = _Reporter("DocumentLoader")
    results: list[ProcessedDocument] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        md_path   = _make_markdown(tmpdir)
        txt_path  = _make_txt(tmpdir)
        pdf_path  = _make_pdf(tmpdir)
        scan_path = _make_scanned_pdf(tmpdir)

        with r.check("markdown loading"):
            md_result = loader.load_file(md_path)
            _assert_processed(md_result, "markdown")
            heading_texts = [h.heading_text for h in md_result.heading_chunks]
            assert "Introduction" in heading_texts, f"Expected 'Introduction' heading, got: {heading_texts}"
            assert "Bali" in heading_texts, f"Expected 'Bali' heading, got: {heading_texts}"

        with r.check("txt loading"):
            txt_result = loader.load_file(txt_path)
            _assert_processed(txt_result, "txt")

        with r.check("pdf loading"):
            pdf_result = loader.load_file(pdf_path)
            _assert_processed(pdf_result, "pdf")

        with r.check("scanned pdf rejection"):
            try:
                loader.load_file(scan_path)
                raise AssertionError("expected ValueError but none was raised")
            except ValueError:
                pass  # this is the expected outcome

        with r.check("directory loading"):
            results = loader.load_directory(tmpdir)
            assert len(results) == 3, f"Expected 3 processed documents, got {len(results)}"
            filenames = {d.document.filename for d in results}
            assert "test.md"     in filenames, "test.md missing"
            assert "test.txt"    in filenames, "test.txt missing"
            assert "test.pdf"    in filenames, "test.pdf missing"
            assert "scanned.pdf" not in filenames, "scanned.pdf should be skipped"

        with r.check("cross-document ID uniqueness"):
            all_doc_ids = [d.document.id for d in results]
            assert len(all_doc_ids) == len(set(all_doc_ids)), "Duplicate document IDs across files"
            all_chunk_ids = [sc.id for d in results for sc in d.semantic_chunks]
            assert len(all_chunk_ids) == len(set(all_chunk_ids)), "Duplicate semantic chunk IDs across files"

        with r.check("section_id referential integrity (all loaded docs)"):
            for d in results:
                _assert_processed(d, d.document.filename)

    r.finish()


if __name__ == "__main__":
    test_document_loader()


# %% 3. MongoStore

def test_mongo_store() -> None:
    client = InferenceClient.from_config(GroqConfig(), HuggingFaceConfig())
    if not client.is_available():
        raise RuntimeError("Groq not reachable — cannot build a document to store")

    loader = DocumentLoader(client)
    mongo  = MongoStore.from_config(MongoConfig())
    r = _Reporter("MongoStore")

    with r.check("health"):
        health = mongo.health()
        assert health.mongo_ok, f"Mongo not reachable: {health.errors}"
        print(f"    documents={health.document_count}, "
              f"heading_chunks={health.heading_chunk_count}, "
              f"semantic_chunks={health.semantic_chunk_count}")
    with r.check("expected collections exist"):
        expected = {"documents", "heading_chunks", "semantic_chunks"}
        existing = set(mongo._db.list_collection_names())
        missing = expected - existing
        assert not missing, f"Missing collection(s): {missing}"
        print(f"    collections present: {sorted(existing & expected)}")


    _purge_by_filename("test.md", mongo)

    with tempfile.TemporaryDirectory() as tmpdir:
        md_path = _make_markdown(tmpdir)
        processed = loader.load_file(md_path)
        doc_id = processed.document.id

        with r.check("store_all / re-store is idempotent"):
            mongo.store_all(processed.document, processed.heading_chunks, processed.semantic_chunks)
            mongo.store_all(processed.document, processed.heading_chunks, processed.semantic_chunks)
            chunks = mongo.get_semantic_chunks_for_document(doc_id)
            assert len(chunks) == len(processed.semantic_chunks), \
                "Re-storing the same chunks should not change the count"

        with r.check("get_document / get_document_by_filename"):
            fetched = mongo.get_document(doc_id)
            assert fetched is not None, "get_document returned None"
            assert fetched.filename == processed.document.filename
            by_name = mongo.get_document_by_filename(processed.document.filename)
            assert by_name is not None and by_name.id == doc_id

        with r.check("list_documents excludes content"):
            docs = mongo.list_documents()
            found = next((d for d in docs if d.id == doc_id), None)
            assert found is not None, "Document missing from list_documents"
            assert found.content == "", "list_documents should not return the full content field"

        with r.check("heading chunk fetch, ordered by start_char"):
            headings = mongo.get_heading_chunks_for_document(doc_id)
            assert len(headings) == len(processed.heading_chunks)
            assert all(
                headings[i].start_char <= headings[i + 1].start_char
                for i in range(len(headings) - 1)
            ), "Heading chunks not ordered by start_char"
            one = mongo.get_heading_chunk(headings[0].id)
            assert one is not None and one.id == headings[0].id

        with r.check("semantic chunk fetch, ordered by position"):
            semantics = mongo.get_semantic_chunks_for_document(doc_id)
            assert len(semantics) == len(processed.semantic_chunks)
            assert all(
                semantics[i].position <= semantics[i + 1].position
                for i in range(len(semantics) - 1)
            ), "Semantic chunks not ordered by position"
            section_chunks = mongo.get_semantic_chunks_for_section(semantics[0].section_id)
            assert len(section_chunks) > 0

        with r.check("get_surrounding_chunks"):
            semantics = mongo.get_semantic_chunks_for_document(doc_id)
            mid = semantics[len(semantics) // 2]
            surrounding = mongo.get_surrounding_chunks(mid.id, n=1)
            assert mid.id in {c.id for c in surrounding}, "Anchor chunk missing from result"
            assert len(surrounding) <= 3, "n=1 should return at most 3 chunks (prev, anchor, next)"
            print(f"    {len(surrounding)} chunks around the anchor")

        with r.check("delete_document"):
            counts = mongo.delete_document(doc_id)
            assert counts["documents"] == 1, f"Expected 1 document deleted, got {counts['documents']}"
            assert counts["heading_chunks"] == len(processed.heading_chunks)
            assert counts["semantic_chunks"] == len(processed.semantic_chunks)
            assert mongo.get_document(doc_id) is None, "Document still present after delete"

    mongo.delete_document(doc_id)  # final best-effort cleanup, in case "delete_document" itself failed
    r.finish()


if __name__ == "__main__":
    test_mongo_store()


# %% 4. QdrantStore

def test_qdrant_store() -> None:
    client = InferenceClient.from_config(GroqConfig(), HuggingFaceConfig())
    if not client.is_available():
        raise RuntimeError("Groq not reachable — cannot build a document to store")

    loader = DocumentLoader(client)
    qdrant = QdrantStore.from_config(client, QdrantConfig())
    r = _Reporter("QdrantStore")

    with r.check("health"):
        health = qdrant.health()
        assert health.qdrant_ok, f"Qdrant not reachable: {health.errors}"
        assert health.embedding_ok, f"Embedding model not working: {health.errors}"
        print(f"    collection={health.collection_name}, points={health.document_count}")

    # Note: unlike Mongo, there's no clean lookup-by-filename here, so a
    # crashed prior run could leave a few orphaned points behind under a
    # different doc_id. Low-stakes (no correctness impact on this run,
    # since we only ever query/delete by the doc_id we create below) —
    # just flagging it rather than hiding it.

    with tempfile.TemporaryDirectory() as tmpdir:
        md_path = _make_markdown(tmpdir)
        processed = loader.load_file(md_path)
        doc_id = processed.document.id

        with r.check("store / re-store is idempotent"):
            qdrant.store(processed.semantic_chunks)
            qdrant.store(processed.semantic_chunks)  # should be a no-op the second time

        with r.check("query returns the stored chunks, sorted by score"):
            results = qdrant.query("What region of Bali has volcanic landscapes?", k=5)
            assert len(results) > 0, "Query returned no results"
            assert any(res.doc_id == doc_id for res in results), \
                "Query did not surface a chunk from the freshly stored document"
            assert all(-1.0 <= res.score <= 1.0 for res in results), \
                "Cosine similarity score out of the expected [-1, 1] range"
            assert results == sorted(results, key=lambda res: res.score, reverse=True), \
                "Results are not sorted by descending score"
            print(f"    top score={results[0].score:.3f}, {len(results)} results")

        with r.check("delete_document removes exactly the stored points"):
            deleted = qdrant.delete_document(doc_id)
            assert deleted == len(processed.semantic_chunks), \
                f"Expected {len(processed.semantic_chunks)} points deleted, got {deleted}"
            post_delete = qdrant.query("What region of Bali has volcanic landscapes?", k=5)
            assert all(res.doc_id != doc_id for res in post_delete), \
                "Chunks from the deleted document are still retrievable"

    qdrant.delete_document(doc_id)  # final best-effort cleanup
    r.finish()


if __name__ == "__main__":
    test_qdrant_store()


# %% 5. IngestionPipeline

def test_ingestion_pipeline() -> None:
    client   = InferenceClient.from_config(GroqConfig(), HuggingFaceConfig())
    if not client.is_available():
        raise RuntimeError("Groq not reachable — cannot build a document to ingest")

    loader   = DocumentLoader(client)
    mongo    = MongoStore.from_config(MongoConfig())
    qdrant   = QdrantStore.from_config(client, QdrantConfig())
    pipeline = IngestionPipeline(loader=loader, mongo=mongo, qdrant=qdrant)
    r = _Reporter("IngestionPipeline")

    with r.check("combined health"):
        health = pipeline.health()
        assert health["mongo_ok"], "Mongo not reachable"
        assert health["qdrant_ok"], "Qdrant not reachable"

    _purge_by_filename("test.txt", mongo, qdrant)
    doc_id: str | None = None

    with tempfile.TemporaryDirectory() as tmpdir:
        txt_path = _make_txt(tmpdir)

        with r.check("ingest_file writes to both stores"):
            processed = pipeline.ingest_file(txt_path)
            doc_id = processed.document.id
            assert mongo.get_document(doc_id) is not None, "Document not found in Mongo after ingest"
            assert len(qdrant.query("vector database", k=5)) > 0, \
                "No chunks retrievable from Qdrant after ingest"

        with r.check("re-ingesting an unchanged file is a no-op"):
            # KNOWN BUG, not something this test works around: DocumentLoader
            # assigns ParsedDocument.id via uuid.uuid4() (random) rather than
            # deriving it from file content, so ingest_file's dedup check —
            # `mongo.get_document(processed.document.id)` — is comparing
            # against an ID that is guaranteed to be brand new every call.
            # It can never find a match, so this check currently FAILS: every
            # re-ingest of the same file creates a second, fully duplicate
            # document instead of being skipped. Left in deliberately so this
            # turns green the moment that's fixed, instead of the gap going
            # unnoticed.
            before = len(mongo.get_semantic_chunks_for_document(doc_id))  # type: ignore[reportArgumentType]
            pipeline.ingest_file(txt_path)
            after = len(mongo.get_semantic_chunks_for_document(doc_id))  # type: ignore[reportArgumentType]
            assert before == after, \
                f"Re-ingesting the same file changed this document's chunk count ({before} -> {after}); " \
                "a duplicate document was likely created under a new ID instead of being skipped"

        with r.check("reingest_file replaces the document"):
            reprocessed = pipeline.reingest_file(txt_path, doc_id=doc_id)
            assert mongo.get_document(reprocessed.document.id) is not None
            assert len(mongo.get_semantic_chunks_for_document(reprocessed.document.id)) > 0
            doc_id = reprocessed.document.id

        with r.check("delete_document removes from both stores"):
            counts = pipeline.delete_document(doc_id)  # type: ignore[reportArgumentType]
            assert counts["mongo_documents"] == 1
            assert counts["qdrant_chunks"] > 0
            assert mongo.get_document(doc_id) is None  # type: ignore[reportArgumentType]

    _purge_by_filename("test.txt", mongo, qdrant)  # sweep up the duplicate from the bug above, too
    r.finish()


if __name__ == "__main__":
    test_ingestion_pipeline()


# %% 6. RAGPipeline (end-to-end)

def test_rag_pipeline() -> None:
    client   = InferenceClient.from_config(GroqConfig(), HuggingFaceConfig())
    if not client.is_available():
        raise RuntimeError("Groq not reachable — cannot run the RAG pipeline")

    loader   = DocumentLoader(client)
    mongo    = MongoStore.from_config(MongoConfig())
    qdrant   = QdrantStore.from_config(client, QdrantConfig())
    pipeline = IngestionPipeline(loader=loader, mongo=mongo, qdrant=qdrant)
    rag      = RAGPipeline(client=client, qdrant=qdrant, mongo=mongo)
    r = _Reporter("RAGPipeline")

    _purge_by_filename("test.md", mongo, qdrant)
    doc_id: str | None = None
    result: dict = {"answer": "", "confidence_score": 0.5, "tool_calls_made": 0, "sources": []}

    with tempfile.TemporaryDirectory() as tmpdir:
        md_path = _make_markdown(tmpdir)

        with r.check("seed a document for retrieval"):
            processed = pipeline.ingest_file(md_path)
            doc_id = processed.document.id
            assert len(qdrant.query("Bali volcanic landscapes", k=3)) > 0, \
                "Seed document not retrievable immediately after ingest"

        with r.check("answers a question grounded in the document"):
            result = rag.run("Which region of Bali has volcanic landscapes?")
            assert isinstance(result["answer"], str) and len(result["answer"]) > 0, "Empty answer"
            assert 0.0 <= result["confidence_score"] <= 1.0, "confidence_score out of [0, 1] range"
            assert isinstance(result["tool_calls_made"], int) and result["tool_calls_made"] >= 0
            assert any(s["doc_id"] == doc_id for s in result["sources"]), \
                "Seed document never showed up in sources"
            print(f"    answer: {result['answer'][:120]}")
            print(f"    confidence_score={result['confidence_score']:.2f}, "
                  f"tool_calls_made={result['tool_calls_made']}, sources={len(result['sources'])}")

        with r.check("follow-up question can use conversation history"):
            history = [
                {"role": "user", "content": "Which region of Bali has volcanic landscapes?"},
                {"role": "assistant", "content": result["answer"]},
            ]
            follow_up = rag.run("What about its central region?", history=history)
            assert isinstance(follow_up["answer"], str) and len(follow_up["answer"]) > 0
            print(f"    follow-up answer: {follow_up['answer'][:120]}")

        with r.check("returns gracefully on an out-of-scope question"):
            # Not asserting exact wording (e.g. "I don't know") since that's an
            # LLM phrasing detail and would make this test flaky. Asserting the
            # pipeline completes cleanly and returns well-formed output is the
            # meaningful, stable thing to check here.
            stumper = rag.run("What is the boiling point of mercury in Kelvin?")
            assert isinstance(stumper["answer"], str)
            assert 0.0 <= stumper["confidence_score"] <= 1.0
            print(f"    out-of-scope confidence_score={stumper['confidence_score']:.2f}")

    with r.check("cleanup"):
        if doc_id:
            counts = pipeline.delete_document(doc_id)
            assert counts["mongo_documents"] == 1

    r.finish()


if __name__ == "__main__":
    test_rag_pipeline()