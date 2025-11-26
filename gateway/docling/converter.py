# gateway/docling/converter.py
from __future__ import annotations
import os
from typing import Dict, Optional, List

from modelscope import snapshot_download

from docling.datamodel.base_models import InputFormat
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    ImageFormatOption,
    WordFormatOption,
    HTMLFormatOption,
    PowerpointFormatOption,
    MarkdownFormatOption,
    AsciiDocFormatOption,
)
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    AcceleratorOptions,
    AcceleratorDevice,
    TableStructureOptions,
    TableFormerMode,
    RapidOcrOptions,
    OcrMacOptions,
)

# Backends
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

from .options import DoclingOptions


# ---------------------------
# module-level singleton cache
# ---------------------------
_converter_singleton: Optional[DocumentConverter] = None
_last_opts_key: Optional[str] = None


# ---------------------------
# helpers
# ---------------------------
def _opts_fingerprint(opts: DoclingOptions) -> str:
    """Build a stable key for caching the converter graph."""
    backends_map = getattr(opts, "backends_map", None) or {}
    parts = [
        str(opts.num_threads or ""),
        str(opts.use_accelerator),
        str(opts.image_scale),
        str(opts.do_ocr),
        ",".join(opts.ocr_langs or []),
        str(opts.do_table_structure),
        str(opts.generate_page_images),
        str(opts.generate_picture_images),
        (opts.table_mode or ""),
        # allowed types
        ",".join(opts.allowed_input_types or []),
        # rapidocr sources
        str(opts.ocr_cache_dir or ""),
        str(opts.rapidocr_repo_id or ""),
        str(opts.rapidocr_det_model_relpath or ""),
        str(opts.rapidocr_rec_model_relpath or ""),
        str(opts.rapidocr_cls_model_relpath or ""),
        # keep a stable representation of backend overrides (even if we don't
        # currently use them for DOCX, this keeps fingerprint future-proof)
        ",".join(f"{k}:{backends_map.get(k, '')}" for k in sorted(backends_map.keys())),
    ]
    return "|".join(parts)


def _resolve_table_mode(mode_str: str | None) -> TableFormerMode:
    m = (mode_str or "").strip().lower()
    return TableFormerMode.FAST if m == "fast" else TableFormerMode.ACCURATE


def _ensure_rapidocr(opts: DoclingOptions) -> tuple[str, str, str]:
    """
    Ensure the RapidOCR bundle is present locally, then return absolute model paths.
    """
    cache_dir = os.path.join(os.path.expanduser("~"), opts.ocr_cache_dir or ".ocr_cache")

    det_path = os.path.join(cache_dir, opts.rapidocr_det_model_relpath)
    rec_path = os.path.join(cache_dir, opts.rapidocr_rec_model_relpath)
    cls_path = os.path.join(cache_dir, opts.rapidocr_cls_model_relpath)

    if not (os.path.exists(det_path) and os.path.exists(rec_path) and os.path.exists(cls_path)):
        snapshot_download(
            repo_id=opts.rapidocr_repo_id or "RapidAI/RapidOCR",
            local_dir=cache_dir,
        )
        # recompute after fetch
        det_path = os.path.join(cache_dir, opts.rapidocr_det_model_relpath)
        rec_path = os.path.join(cache_dir, opts.rapidocr_rec_model_relpath)
        cls_path = os.path.join(cache_dir, opts.rapidocr_cls_model_relpath)

    return det_path, rec_path, cls_path


def _build_pdf_pipeline_options(opts: DoclingOptions) -> PdfPipelineOptions:
    """
    Translate DoclingOptions into PdfPipelineOptions (images, ocr, tables, threading).
    """
    pdf_opts = PdfPipelineOptions()

    # threading / accelerator
    pdf_opts.accelerator_options = AcceleratorOptions(
        num_threads=max(1, int(opts.num_threads or 1)),
        device=AcceleratorDevice.AUTO if opts.use_accelerator else AcceleratorDevice.CPU,
    )

    # imaging
    pdf_opts.images_scale = float(opts.image_scale)
    pdf_opts.generate_page_images = bool(opts.generate_page_images)
    pdf_opts.generate_picture_images = bool(opts.generate_picture_images)

    # OCR
    if opts.do_ocr:
        det, rec, cls = _ensure_rapidocr(opts)
        pdf_opts.do_ocr = True
        pdf_opts.ocr_options = RapidOcrOptions(
            lang=opts.ocr_langs or ["english"],
            force_full_page_ocr=False,
            det_model_path=det,
            rec_model_path=rec,
            cls_model_path=cls,
        )
    else:
        pdf_opts.do_ocr = False

    # Tables
    pdf_opts.do_table_structure = bool(opts.do_table_structure)
    if pdf_opts.do_table_structure:
        pdf_opts.table_structure_options = TableStructureOptions(
            do_cell_matching=opts.do_cell_matching,
            mode=_resolve_table_mode(opts.table_mode),
        )

    return pdf_opts


def _allowed_formats_from_opts(opts: DoclingOptions) -> List[InputFormat]:
    """
    Convert string names to InputFormat enums.
    IMPORTANT: Do NOT include CSV by default unless you also add a CSV format option.
    """
    default_names = ["pdf", "image", "docx", "html", "pptx", "asciidoc", "md"]
    names = [n.lower() for n in (opts.allowed_input_types or default_names)]

    mapping = {
        "pdf": InputFormat.PDF,
        "image": InputFormat.IMAGE,
        "docx": InputFormat.DOCX,
        "html": InputFormat.HTML,
        "pptx": InputFormat.PPTX,
        "asciidoc": InputFormat.ASCIIDOC,
        "md": InputFormat.MD,
        "markdown": InputFormat.MD,
        # "csv": InputFormat.CSV,  # enable only if you add a CSV format option below
    }
    out: List[InputFormat] = []
    for n in names:
        fmt = mapping.get(n)
        if fmt is not None:
            out.append(fmt)
    return out


def _format_options_for_allowed_types(
    pdf_opts: PdfPipelineOptions,
    allowed: List[InputFormat],
) -> Dict[InputFormat, object]:
    """
    Build mapping from InputFormat -> *FormatOption.
    Use the specialized *FormatOption classes without forcing pipeline_cls/backend.
    """
    m: Dict[InputFormat, object] = {}

    if InputFormat.PDF in allowed:
        m[InputFormat.PDF] = PdfFormatOption(
            backend=DoclingParseV2DocumentBackend,
            pipeline_cls=StandardPdfPipeline,
            pipeline_options=pdf_opts,
        )

    if InputFormat.IMAGE in allowed:
        # Images are handled by the same PDF pipeline under the hood; most versions accept no args here.
        m[InputFormat.IMAGE] = ImageFormatOption()

    if InputFormat.DOCX in allowed:
        m[InputFormat.DOCX] = WordFormatOption()

    if InputFormat.HTML in allowed:
        m[InputFormat.HTML] = HTMLFormatOption()

    if InputFormat.PPTX in allowed:
        m[InputFormat.PPTX] = PowerpointFormatOption()

    if InputFormat.MD in allowed:
        m[InputFormat.MD] = MarkdownFormatOption()

    if InputFormat.ASCIIDOC in allowed:
        m[InputFormat.ASCIIDOC] = AsciiDocFormatOption()

    # If you ever enable CSV in allowed formats, be sure to provide an option here too.

    return m


def _build_converter(opts: DoclingOptions) -> DocumentConverter:
    """Construct a DocumentConverter with explicit allowed_formats + per-format options."""
    pdf_opts = _build_pdf_pipeline_options(opts)
    allowed = _allowed_formats_from_opts(opts)
    fmt_opts = _format_options_for_allowed_types(pdf_opts, allowed)

    # Sanity: ensure each allowed format has an option
    missing = [f for f in allowed if f not in fmt_opts]
    if missing:
        raise ValueError(f"Missing format_options for: {missing}")

    return DocumentConverter(
        allowed_formats=allowed,
        format_options=fmt_opts,
    )


# ---------------------------
# public API
# ---------------------------
def get_converter(opts: DoclingOptions, *, fresh: bool = False) -> DocumentConverter:
    """
    Return a cached DocumentConverter keyed by a fingerprint of DoclingOptions.
    Set fresh=True to force rebuild.
    """
    global _converter_singleton, _last_opts_key
    key = _opts_fingerprint(opts)

    if fresh or _converter_singleton is None or key != _last_opts_key:
        _converter_singleton = _build_converter(opts)
        _last_opts_key = key

    return _converter_singleton
    
    # pipeline_options = PdfPipelineOptions(
    #     do_ocr=True,                       # ← keep False for now
    #     images_scale=2.0,                   # ↓ raster size
    #     generate_page_images=True,         # off = less memory
    #     generate_picture_images=True,      # off = less memory
    #     do_table_structure=True,           # off = less memory
    #     accelerator_options=AcceleratorOptions(
    #         num_threads=1,                  # conservative
    #         device=AcceleratorDevice.MPS,   # ← force CPU (avoids MPS)
    #     ),
    #     # ocr_options=ocr_options,
    # )
    
    # converter = DocumentConverter(
    #     format_options={
    #         InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
    #     },
    # )
    
    # return converter
