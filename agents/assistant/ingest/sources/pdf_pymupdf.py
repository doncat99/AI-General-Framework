from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Union
from pathlib import Path

from loguru import logger
import fitz  # type: ignore


@dataclass
class PDFChunk:
    text: str
    metadata: Dict[str, Union[str, int, float]]


class PDFPyMuPDFSource:
    """
    Simple PDF → text chunk source using PyMuPDF (fitz).

    - Accepts a single PDF file path or a directory (recurses for *.pdf).
    - Emits dicts with keys: "text", "metadata".
    - Chunking is by character length with optional overlap.
    """

    def __init__(
        self,
        input_path: Union[str, Path],
        *,
        chunk_chars: int = 1200,
        chunk_overlap: int = 120,
        min_chunk_chars: int = 200,
        recursive: bool = True,
        encoding_hint: Optional[str] = None,  # kept for API parity; PyMuPDF handles text decoding internally
    ) -> None:
        self.input_path = Path(input_path)
        self.chunk_chars = max(1, int(chunk_chars))
        self.chunk_overlap = max(0, int(chunk_overlap))
        self.min_chunk_chars = max(1, int(min_chunk_chars))
        self.recursive = recursive
        self.encoding_hint = encoding_hint

        if not self.input_path.exists():
            raise FileNotFoundError(f"PDF input path not found: {self.input_path}")

    # Public API used by IngestPipeline
    def iter_docs(self) -> Iterator[Dict[str, object]]:
        for pdf_path in self._iter_pdf_files():
            yield from self._emit_pdf(pdf_path)

    # Also allow simple iteration
    def __iter__(self) -> Iterator[Dict[str, object]]:
        return self.iter_docs()

    # ------------- internals -------------

    def _iter_pdf_files(self) -> Iterable[Path]:
        if self.input_path.is_file():
            if self.input_path.suffix.lower() == ".pdf":
                yield self.input_path
            else:
                logger.warning(f"Skipping non-PDF file: {self.input_path}")
            return

        pattern = "**/*.pdf" if self.recursive else "*.pdf"
        count = 0
        for p in self.input_path.glob(pattern):
            count += 1
            yield p
        if count == 0:
            logger.warning(f"No PDF files found under: {self.input_path}")

    def _emit_pdf(self, pdf_path: Path) -> Iterator[Dict[str, object]]:
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            logger.error(f"Failed to open PDF '{pdf_path}': {e}")
            return

        with doc:
            for page_index in range(doc.page_count):
                try:
                    page = doc.load_page(page_index)
                    # "text" is the default layout-aware extractor
                    raw = page.get_text("text") or ""
                    text = raw.strip()
                except Exception as e:
                    logger.warning(f"Failed to read page {page_index+1} of {pdf_path.name}: {e}")
                    continue

                if not text:
                    continue

                base_meta = {
                    "source": str(pdf_path),
                    "filename": pdf_path.name,
                    "page": page_index + 1,
                    "doc_id": pdf_path.stem,
                }

                for i, chunk in enumerate(self._chunk_text(text)):
                    if len(chunk) < self.min_chunk_chars:
                        continue
                    meta = dict(base_meta)
                    meta["chunk"] = i
                    yield {"text": chunk, "metadata": meta}

    def _chunk_text(self, text: str) -> List[str]:
        """
        Greedy paragraph packer with char budget + overlap.
        """
        paras = [p.strip() for p in text.splitlines() if p.strip()]
        if not paras:
            return []

        chunks: List[str] = []
        buf: List[str] = []
        buf_len = 0

        def flush():
            if buf:
                chunks.append(" ".join(buf).strip())

        for p in paras:
            if buf_len + len(p) + (1 if buf else 0) <= self.chunk_chars:
                buf.append(p)
                buf_len += len(p) + (1 if buf_len > 0 else 0)
            else:
                flush()
                # build next buffer with overlap
                if self.chunk_overlap > 0 and chunks:
                    # take tail from last chunk
                    tail = chunks[-1][-self.chunk_overlap :]
                    # don't split mid-word
                    tail = tail[tail.find(" ") + 1 :] if " " in tail else tail
                    buf = [tail, p] if tail else [p]
                    buf_len = len(" ".join(buf))
                else:
                    buf = [p]
                    buf_len = len(p)

        flush()
        return chunks
