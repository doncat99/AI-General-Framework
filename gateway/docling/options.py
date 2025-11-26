# options.py
from __future__ import annotations

import os
from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class DoclingOptions(BaseModel):
    # ---------------------------
    # Core toggles
    # ---------------------------
    do_ocr: bool = Field(default=True, description="Enable OCR for image/PDF text.")
    ocr_langs: List[str] = Field(
        default_factory=lambda: ["english"],
        description="OCR language codes passed to RapidOCR.",
    )
    do_table_structure: bool = Field(
        default=True, description="Enable table structure recovery."
    )
    do_cell_matching: bool = Field(
        default=True, description="Enable table cell matching step."
    )
    image_scale: float = Field(
        default=1.0,
        ge=0.5,
        le=6.0,
        description="Scale factor for rasterized page/picture images.",
    )
    generate_page_images: bool = Field(
        default=True, description="Emit page-level images during conversion."
    )
    generate_picture_images: bool = Field(
        default=True, description="Emit extracted picture images during conversion."
    )

    # ---------------------------
    # Pipeline selection / parsing
    # ---------------------------
    # Optional per-format backend mapping (e.g., {'pdf': 'pypdfium2'})
    backends_map: Optional[Dict[str, str]] = Field(
        default=None,
        description="Optional per-format backend mapping. Keys like 'pdf', 'image', 'docx', etc.",
    )

    table_mode: str = Field(
        default="accurate",
        description="TableFormer mode: 'accurate' or 'fast' (converter maps to enum).",
    )

    # Page constraints (optional). If both None, process all pages.
    page_start: Optional[int] = Field(
        default=None, ge=1, description="1-based start page (inclusive)."
    )
    page_end: Optional[int] = Field(
        default=None, ge=1, description="1-based end page (inclusive)."
    )

    # Input allowance (used by converter to configure allowed formats)
    allowed_input_types: List[str] = Field(
        default_factory=lambda: ["pdf", "image", "docx", "html", "pptx", "asciidoc", "md"],
        description="Human-readable list; converter maps to docling InputFormat.",
    )

    # ---------------------------
    # Accelerator & threading
    # ---------------------------
    num_threads: Optional[int] = Field(
        default=1, ge=1, description="Limit CPU threads used inside pipeline."
    )
    use_accelerator: bool = Field(
        default=True,
        description="If true, use AcceleratorOptions (AUTO device) when available.",
    )

    # ---------------------------
    # OCR model sources (optional)
    # ---------------------------
    ocr_cache_dir: Optional[str] = Field(
        # default=".cache/modelscope/hub/models/RapidAI/RapidOCR",
        default=".ocr_cache",
        description="Local cache directory for RapidOCR models.",
    )
    rapidocr_repo_id: Optional[str] = Field(
        default="RapidAI/RapidOCR",
        description="HuggingFace repo id for RapidOCR models.",
    )
    rapidocr_det_model_relpath: Optional[str] = Field(
        default="onnx/PP-OCRv5/det/ch_PP-OCRv5_server_det.onnx",
        description="Relative path (under cache dir) to detector model.",
    )
    rapidocr_rec_model_relpath: Optional[str] = Field(
        default="onnx/PP-OCRv5/rec/ch_PP-OCRv5_rec_server_infer.onnx",
        description="Relative path (under cache dir) to recognizer model.",
    )
    rapidocr_cls_model_relpath: Optional[str] = Field(
        default="onnx/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_infer.onnx",
        description="Relative path (under cache dir) to classifier model.",
    )

    # ---------------------------
    # Post-processors (Header/Footer + Label correction)
    # ---------------------------
    enable_header_footer_detection: bool = Field(
        default=False, description="Enable header/footer relabel pass."
    )
    enable_label_correction: bool = Field(
        default=True, description="Run label-correction heuristics."
    )
    label_merge_distance_px: float = Field(
        default=50.0, description="Vertical distance for merging adjacent headers."
    )
    label_min_header_score: float = Field(
        default=4.5, description="Minimum score to keep a title/section header label."
    )

    # ---------------------------
    # Exporters
    # ---------------------------
    export_markdown: bool = Field(default=True)
    export_doc_json: bool = Field(default=True)
    export_deep_json_report: bool = Field(
        default=True, description="Structured deep report (DeepJsonReportExporter)."
    )
    export_domtree_json: bool = Field(
        default=False, description="DomTreeExporter JSON (structured content)."
    )

    # DomTree serialization fine-tuning
    domtree_strict_text: bool = Field(default=False)
    domtree_indent: int = Field(default=4)
    domtree_text_width: int = Field(default=-1)
    domtree_merge_headers: bool = Field(default=True)
    domtree_merge_distance: float = Field(default=6.5)
    # IMPORTANT: default changed to **artifacts** (capitalized) per your prior convention
    domtree_artifacts_folder: str = Field(
        default="artifacts",
        description="Folder for exported images/tables if any exporter needs it.",
    )

    # ---------------------------
    # Storage
    # ---------------------------
    base_dir: str = Field(
        default=".cache/docling",
        description="Root directory for LocalStorage persistence.",
    )
    overwrite_existing: bool = Field(
        default=True, description="If true, stored files may be overwritten."
    )
    keep_payload_in_memory: bool = Field(
        default=False,
        description="Also keep bytes in Artifact.payload after persisting.",
    )

    # ---------------------------
    # Batch / runtime controls
    # ---------------------------
    fail_fast: bool = Field(
        default=False,
        description="If true, raise on first conversion error in batch.",
    )
    parallel_workers: Optional[int] = Field(
        default=None, ge=1, description="Optional parallelism for batch; None = serial."
    )
    log_level: str = Field(default="INFO", description="String log level.")

    class Config:
        arbitrary_types_allowed = True
