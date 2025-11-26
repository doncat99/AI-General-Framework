from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional


# -----------------------------------------------------------------------------
# Artifact taxonomy (kept from your previous version)
# -----------------------------------------------------------------------------
class ArtifactType(str, Enum):
    MARKDOWN = "markdown"
    DOC_JSON = "doc_json"
    JSON = "json"
    DOMTREE_JSON = "domtree_json"
    OTHER = "other"


@dataclass
class Artifact:
    """
    A single produced file/blob from the pipeline.
    - path: where it was persisted (if persisted)
    - name: logical filename (used when saving if provided)
    - payload: in-memory bytes (when not persisted yet or for MemoryStorage)
    - meta: free-form metadata (page_no, table_id, content-type, etc.)
    """
    type: ArtifactType
    name: str
    payload: Optional[bytes] = None
    path: Optional[Path] = None


class ResultStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class Result:
    status: ResultStatus
    input_name: str
    artifacts: List[Artifact] = field(default_factory=list)
    error: Optional[str] = None
    meta: Dict[str, object] = field(default_factory=dict)


@dataclass
class BatchManifest:
    results: List[Result] = field(default_factory=list)
