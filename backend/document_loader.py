import re
import uuid
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fitz  # pymupdf
from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import SemanticChunker
from pydantic import BaseModel
from tqdm import tqdm
from inference_client import InferenceClient

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCANNED_PDF_CHAR_THRESHOLD = 500
SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt"}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ParsedDocument(BaseModel):
    id: str
    filename: str
    filepath: str
    filetype: str
    content: str = ""
    char_count: int
    metadata: dict[str, Any] = {}


class HeadingChunk(BaseModel):
    id: str
    doc_id: str
    heading_text: str
    level: int  # 1 = h1, 2 = h2, 3 = h3, 0 = implicit (no heading found)
    content: str  # full text under this heading
    start_char: int
    end_char: int


class SemanticChunk(BaseModel):
    id: str
    doc_id: str
    section_id: str  # → HeadingChunk.id
    content: str
    position: int  # order within document
    prev_id: str | None = None
    next_id: str | None = None
    metadata: dict[str, Any] = {}


class ProcessedDocument(BaseModel):
    document: ParsedDocument
    heading_chunks: list[HeadingChunk]
    semantic_chunks: list[SemanticChunk]


# ---------------------------------------------------------------------------
# LangChain embeddings adapter
# ---------------------------------------------------------------------------


class _OllamaLangChainEmbeddings(Embeddings):
    """
    Thin adapter so SemanticChunker can use OllamaClient.embed
    without pulling in a separate LangChain-Ollama dependency.
    """
    def __init__(self, client: InferenceClient) -> None:
        self._client = client

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._client.embed_batch(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._client.embed(text)

class _InferenceLangChainEmbeddings(Embeddings):
    def __init__(self, client: InferenceClient) -> None:
        self._client = client

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._client.embed_batch(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._client.embed(text)

# ---------------------------------------------------------------------------
# Internal dataclass (not exposed publicly)
# ---------------------------------------------------------------------------


@dataclass
class _RawHeading:
    """Intermediate heading representation before IDs are assigned."""

    heading_text: str
    level: int
    start_char: int
    end_char: int = 0  # filled in after all headings are collected
    content: str = ""  # filled in after slicing full text


# ---------------------------------------------------------------------------
# DocumentLoader
# ---------------------------------------------------------------------------


class DocumentLoader:
    """
    Loads documents from a directory, extracts text, detects heading sections,
    and produces semantic chunks — all without touching any database.

    Usage:
        loader = DocumentLoader(ollama_client)
        results = loader.load_directory("./docs") # results: list[ProcessedDocument]
    """

    def __init__(self, client: InferenceClient) -> None:
        self._client = client
        self._embeddings = _InferenceLangChainEmbeddings(client)
        self._chunker = SemanticChunker(
            embeddings=self._embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=85,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_directory(self, directory: str) -> list[ProcessedDocument]:
        """
        Scan a directory for supported files, process each one, and
        print a progress summary to stdout.

        Returns only successfully processed documents.
        """
        dirpath = Path(directory)
        if not dirpath.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        files = [
            f
            for f in sorted(dirpath.iterdir())
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        if not files:
            print(f"No supported files found in {directory}")
            return []

        total = len(files)
        print(f"\nLoading documents from: {dirpath.resolve()}")
        print("─" * 50)

        results: list[ProcessedDocument] = []
        skipped = 0

        for i, filepath in enumerate(files, start=1):
            prefix = f"[{i}/{total}] {filepath.name:<30}"
            try:
                processed = self._process_file(filepath)
                if processed is None:
                    # Scanned PDF — warning already printed inside _process_file
                    skipped += 1
                    continue

                n_chunks = len(processed.semantic_chunks)
                n_sections = len(processed.heading_chunks)
                print(f"{prefix} ✓  {n_chunks} chunks, {n_sections} sections")
                results.append(processed)

            except Exception as exc:
                print(f"{prefix} ✗  ERROR: {exc}")
                skipped += 1

        print("─" * 50)
        print(f"Done. {len(results)} document(s) processed, {skipped} skipped.\n")
        return results

    def load_file(self, filepath: str) -> ProcessedDocument:
        """Process a single file. Raises on scanned PDFs or unsupported types."""
        result = self._process_file(Path(filepath))
        if result is None:
            raise ValueError(
                f"{filepath} appears to be a scanned PDF "
                f"(fewer than {SCANNED_PDF_CHAR_THRESHOLD} characters extracted)."
            )
        return result

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

    def _process_file(self, filepath: Path) -> ProcessedDocument | None:
        ext = filepath.suffix.lower()

        # 1. Extract text
        if ext == ".pdf":
            parsed = self._parse_pdf(filepath)
        elif ext == ".md":
            parsed = self._parse_markdown(filepath)
        elif ext == ".txt":
            parsed = self._parse_txt(filepath)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        if parsed is None:
            return None  # scanned PDF

        # 2. Detect heading chunks (Pass 1)
        heading_chunks = self._extract_heading_chunks(parsed, ext)

        # 3. Semantic chunking (Pass 2)
        raw_semantic_texts: list[str] = []
        paragraphs = [p.strip() for p in parsed.content.split("\n\n") if p.strip()]
        for para in tqdm(paragraphs, desc=f"chunking {parsed.filename}", unit="para"):
            if len(para) < 50:
                raw_semantic_texts.append(para)
            else:
                try:
                    chunks = self._chunker.split_text(para)
                    raw_semantic_texts.extend(chunks)
                except Exception:
                    sentences = para.replace(". ", ".\n").split("\n")
                    raw_semantic_texts.extend(s.strip() for s in sentences if s.strip())

        # 4. Build SemanticChunk objects, link prev/next, assign section_id
        semantic_chunks = self._build_semantic_chunks(
            raw_semantic_texts, parsed, heading_chunks
        )

        return ProcessedDocument(
            document=parsed,
            heading_chunks=heading_chunks,
            semantic_chunks=semantic_chunks,
        )

    # ------------------------------------------------------------------
    # Text extraction
    # ------------------------------------------------------------------

    def _parse_pdf(self, filepath: Path) -> ParsedDocument | None:
        doc = fitz.open(str(filepath))
        pages: list[str] = []

        for page in doc:
            pages.append(page.get_text())  # type: ignore[attr-defined]

        doc.close()
        content = "\n".join(pages).strip()

        if len(content) < SCANNED_PDF_CHAR_THRESHOLD:
            warnings.warn(
                f"⚠  {filepath.name}: only {len(content)} characters extracted. "
                "This is likely a scanned PDF. Skipping. "
                "(OCR support can be added later.)",
                UserWarning,
                stacklevel=3,
            )
            print(
                f"{'':30} ⚠  WARNING: only {len(content)} chars detected, "
                "likely scanned. Skipping."
            )
            return None

        return ParsedDocument(
            id=str(uuid.uuid4()),
            filename=filepath.name,
            filepath=str(filepath.resolve()),
            filetype="pdf",
            content=content,
            char_count=len(content),
        )

    def _parse_markdown(self, filepath: Path) -> ParsedDocument:
        content = filepath.read_text(encoding="utf-8").strip()
        return ParsedDocument(
            id=str(uuid.uuid4()),
            filename=filepath.name,
            filepath=str(filepath.resolve()),
            filetype="md",
            content=content,
            char_count=len(content),
        )

    def _parse_txt(self, filepath: Path) -> ParsedDocument:
        content = filepath.read_text(encoding="utf-8").strip()
        return ParsedDocument(
            id=str(uuid.uuid4()),
            filename=filepath.name,
            filepath=str(filepath.resolve()),
            filetype="txt",
            content=content,
            char_count=len(content),
        )

    # ------------------------------------------------------------------
    # Heading detection (Pass 1)
    # ------------------------------------------------------------------

    def _extract_heading_chunks(
        self, parsed: ParsedDocument, ext: str
    ) -> list[HeadingChunk]:
        if ext == ".md":
            raw = self._detect_headings_markdown(parsed.content)
        elif ext == ".pdf":
            raw = self._detect_headings_pdf(parsed.filepath, parsed.content)
        else:
            raw = self._detect_headings_txt(parsed.content)

        # If no headings found, treat the entire document as one implicit section
        if not raw:
            return [
                HeadingChunk(
                    id=str(uuid.uuid4()),
                    doc_id=parsed.id,
                    heading_text="[Document]",
                    level=0,
                    content=parsed.content,
                    start_char=0,
                    end_char=len(parsed.content),
                )
            ]

        # Fill in end_char and content for each heading
        chunks: list[HeadingChunk] = []
        for i, h in enumerate(raw):
            end_char = (
                raw[i + 1].start_char if i + 1 < len(raw) else len(parsed.content)
            )
            content = parsed.content[h.start_char : end_char].strip()
            chunks.append(
                HeadingChunk(
                    id=str(uuid.uuid4()),
                    doc_id=parsed.id,
                    heading_text=h.heading_text,
                    level=h.level,
                    content=content,
                    start_char=h.start_char,
                    end_char=end_char,
                )
            )

        return chunks

    def _detect_headings_markdown(self, content: str) -> list[_RawHeading]:
        """Detect ATX-style markdown headings (# / ## / ###)."""
        headings: list[_RawHeading] = []
        for match in re.finditer(r"^(#{1,3})\s+(.+)$", content, re.MULTILINE):
            level = len(match.group(1))
            text = match.group(2).strip()
            headings.append(
                _RawHeading(
                    heading_text=text,
                    level=level,
                    start_char=match.start(),
                )
            )
        return headings

    def _detect_headings_pdf(self, filepath: str, content: str) -> list[_RawHeading]:
        """
        Use pymupdf font-size and bold flags to detect headings.
        Falls back to empty list if the PDF has no structural hints.
        """
        doc = fitz.open(filepath)
        headings: list[_RawHeading] = []

        def _get_spans(fitz_doc: fitz.Document) -> list[dict[str, Any]]:
            """Flatten all spans across all pages into plain dicts."""
            result: list[dict[str, Any]] = []
            for p in fitz_doc:
                page_dict: dict[str, Any] = p.get_text("dict")  # type: ignore[assignment]
                for block in page_dict.get("blocks", []):
                    if not isinstance(block, dict) or block.get("type") != 0:
                        continue
                    for line in block.get("lines", []):
                        if not isinstance(line, dict):
                            continue
                        for span in line.get("spans", []):
                            if isinstance(span, dict):
                                result.append(span)
            return result

        def _get_lines(
            fitz_doc: fitz.Document,
        ) -> list[tuple[str, list[dict[str, Any]]]]:
            """Flatten all lines as (line_text, spans) tuples."""
            result: list[tuple[str, list[dict[str, Any]]]] = []
            for p in fitz_doc:
                page_dict: dict[str, Any] = p.get_text("dict")  # type: ignore[assignment]
                for block in page_dict.get("blocks", []):
                    if not isinstance(block, dict) or block.get("type") != 0:
                        continue
                    for line in block.get("lines", []):
                        if not isinstance(line, dict):
                            continue
                        spans: list[dict[str, Any]] = [
                            s for s in line.get("spans", []) if isinstance(s, dict)
                        ]
                        line_text = " ".join(
                            str(s.get("text", "")) for s in spans
                        ).strip()
                        result.append((line_text, spans))
            return result

        # Pass 1: collect font sizes to find median body size
        font_sizes: list[float] = [
            float(span["size"]) for span in _get_spans(doc) if "size" in span
        ]

        if not font_sizes:
            doc.close()
            return []

        body_size: float = sorted(font_sizes)[len(font_sizes) // 2]

        # Pass 2: flag lines larger/bolder than body as headings
        seen_texts: set[str] = set()
        for line_text, spans in _get_lines(doc):
            if not line_text or line_text in seen_texts:
                continue

            avg_size = sum(float(s.get("size", body_size)) for s in spans) / len(spans)
            is_bold = any("bold" in str(s.get("font", "")).lower() for s in spans)

            if avg_size >= body_size * 1.2 or (is_bold and avg_size >= body_size):
                ratio = avg_size / body_size
                level = 1 if ratio >= 1.5 else (2 if ratio >= 1.25 else 3)

                idx = content.find(line_text)
                if idx == -1:
                    continue

                seen_texts.add(line_text)
                headings.append(
                    _RawHeading(
                        heading_text=line_text,
                        level=level,
                        start_char=idx,
                    )
                )

        doc.close()

        # Sort by position in document text
        headings.sort(key=lambda h: h.start_char)
        return headings

    def _detect_headings_txt(self, content: str) -> list[_RawHeading]:
        """
        Heuristic heading detection for plain text:
        - Lines followed by === (h1) or --- (h2)  [setext style]
        - ALL CAPS lines (h2)
        - Lines ending with a colon that are short (h3)
        """
        headings: list[_RawHeading] = []
        lines = content.split("\n")
        char_offset = 0

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Setext-style: next line is all = or all -
            if i + 1 < len(lines):
                next_stripped = lines[i + 1].strip()
                if next_stripped and all(c == "=" for c in next_stripped) and stripped:
                    headings.append(
                        _RawHeading(
                            heading_text=stripped, level=1, start_char=char_offset
                        )
                    )
                    char_offset += len(line) + 1
                    char_offset += len(lines[i + 1]) + 1
                    i += 2
                    continue
                if next_stripped and all(c == "-" for c in next_stripped) and stripped:
                    headings.append(
                        _RawHeading(
                            heading_text=stripped, level=2, start_char=char_offset
                        )
                    )
                    char_offset += len(line) + 1
                    char_offset += len(lines[i + 1]) + 1
                    i += 2
                    continue

            # ALL CAPS line (at least 4 chars, not a separator)
            if (
                stripped
                and stripped == stripped.upper()
                and len(stripped) >= 4
                and not all(c in "-=_* " for c in stripped)
            ):
                headings.append(
                    _RawHeading(heading_text=stripped, level=2, start_char=char_offset)
                )

            char_offset += len(line) + 1
            i += 1

        return headings

    # ------------------------------------------------------------------
    # Semantic chunk building (Pass 2 + 3)
    # ------------------------------------------------------------------

    def _build_semantic_chunks(
        self,
        raw_texts: list[str],
        parsed: ParsedDocument,
        heading_chunks: list[HeadingChunk],
    ) -> list[SemanticChunk]:
        """
        Turn raw text segments from SemanticChunker into SemanticChunk objects,
        assigning section_id by matching each chunk's position in the document
        to the correct HeadingChunk range.
        """
        chunks: list[SemanticChunk] = []
        search_start = 0

        for position, text in enumerate(raw_texts):
            chunk_id = str(uuid.uuid4())

            # Find approximate char offset of this chunk in the full document
            idx = parsed.content.find(text[:80], search_start)
            if idx == -1:
                # Fallback: scan without offset constraint
                idx = parsed.content.find(text[:80])
            if idx == -1:
                idx = search_start  # give up, use current position
            else:
                search_start = idx + len(text)

            section_id = self._find_section_id(idx, heading_chunks)

            chunks.append(
                SemanticChunk(
                    id=chunk_id,
                    doc_id=parsed.id,
                    section_id=section_id,
                    content=text,
                    position=position,
                    metadata={"source": parsed.filename},
                )
            )

        # Link prev/next
        for i, chunk in enumerate(chunks):
            chunk.prev_id = chunks[i - 1].id if i > 0 else None
            chunk.next_id = chunks[i + 1].id if i + 1 < len(chunks) else None

        return chunks

    def _find_section_id(
        self, char_offset: int, heading_chunks: list[HeadingChunk]
    ) -> str:
        """Return the id of the HeadingChunk whose range contains char_offset."""
        if not heading_chunks:
            return ""
        best = heading_chunks[0]

        for hc in heading_chunks:
            if hc.start_char <= char_offset:
                best = hc
            else:
                break
        return best.id
