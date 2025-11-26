# gateway/docling/__init__.py
from __future__ import annotations

# Options / settings
from .options import DoclingOptions

# Converter factory (uses YAML overrides transparently via apply_settings_overrides)
from .converter import get_converter

# Types & storage
from .types import Artifact, ArtifactType, Result, ResultStatus, BatchManifest
from .storage import StorageBackend, LocalStorage, MemoryStorage

# Post-processors & exporters
from .plugins import (
    PostProcessor,
    HeaderFooterDetectorProcessor,
    LabelCorrectionProcessor,
    JsonExporter,
    MarkdownExporter,
    DomTreeExporter,
    DeepJsonReportExporter,
)

# Optional pipeline façade (if you expose it)
from .settings import (
    get_settings,
    apply_settings_overrides,
    want_processors,
    SettingsManager,
)
from .service import DoclingService

__all__ = [

    # options / settings
    "DoclingOptions",

    # converter
    "get_converter",

    # io/types
    "Artifact",
    "ArtifactType",
    "Result",
    "ResultStatus",
    "BatchManifest",

    # storage
    "StorageBackend",
    "LocalStorage",
    "MemoryStorage",

    # processors / exporters
    "PostProcessor",
    "HeaderFooterDetectorProcessor",
    "LabelCorrectionProcessor",
    "JsonExporter",
    "MarkdownExporter",
    "DomTreeExporter",
    "DeepJsonReportExporter",

    # settings / hot reload
    "get_settings",
    "apply_settings_overrides",
    "want_processors",
    "SettingsManager",
    
    # service
    "DoclingService",
]
