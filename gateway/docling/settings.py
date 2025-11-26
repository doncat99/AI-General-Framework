# settings.py
from __future__ import annotations

import os
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from .options import DoclingOptions

DEFAULT_SETTINGS_BASENAME = "docling_settings.yaml"
ENV_SETTINGS_PATH = "DOCLING_SETTINGS"


# ------------------------------
# Hot-reload state
# ------------------------------
@dataclass
class SettingsState:
    path: Optional[Path] = None
    mtime: float = 0.0
    data: Dict[str, Any] = field(default_factory=dict)


class SettingsManager:
    """
    Thread-safe YAML settings with hot reload (mtime-based).
    """
    _instance = None
    _lock = threading.RLock()

    def __init__(self):
        self._state = SettingsState()

    @classmethod
    def instance(cls) -> "SettingsManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = SettingsManager()
        return cls._instance

    def _resolve_path(self, hint: Optional[str] = None) -> Optional[Path]:
        # explicit hint first
        if hint:
            p = Path(hint).expanduser()
            if p.is_file():
                return p

        # environment
        env = os.getenv(ENV_SETTINGS_PATH, "").strip()
        if env:
            p = Path(env).expanduser()
            if p.is_file():
                return p

        # cwd
        p = Path.cwd() / DEFAULT_SETTINGS_BASENAME
        if p.is_file():
            return p

        # next to this file
        p = Path(__file__).resolve().parent / DEFAULT_SETTINGS_BASENAME
        return p if p.is_file() else None

    def _needs_reload(self, path: Path) -> bool:
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            return False
        return abs(mtime - self._state.mtime) > 1e-6

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        try:
            with path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def get(self, hint: Optional[str] = None) -> Tuple[Dict[str, Any], Optional[Path]]:
        with self._lock:
            p = self._resolve_path(hint)
            if p is None:
                # no file -> clear cached state
                self._state = SettingsState()
                return {}, None

            if self._state.path is None or self._state.path != p or self._needs_reload(p):
                data = self._load_yaml(p)
                try:
                    self._state.mtime = p.stat().st_mtime
                except Exception:
                    self._state.mtime = time.time()
                self._state.path = p
                self._state.data = data

            return dict(self._state.data), self._state.path


# ------------------------------
# Normalization / loading helpers
# ------------------------------
def _apply_aliases(settings: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize yaml keys so downstream code can populate DoclingOptions consistently.

    Priority rules:
      - domtree_artifacts_folder
    """
    s = dict(settings)

    # artifacts folder (single source of truth in options)
    art = s.get("domtree_artifacts_folder")
    if art:
        s["domtree_artifacts_folder"] = str(art)

    # broadly useful defaults
    s.setdefault("num_threads", 1)                 # <- default 1 as requested
    s.setdefault("label_min_header_score", 4.5)    # used by LabelCorrectionProcessor in your stack
    s.setdefault("label_merge_distance_px", 50.0)  # header merge distance for DOM exporters

    # ensure list types are lists
    if "allowed_input_types" in s and not isinstance(s["allowed_input_types"], list):
        s["allowed_input_types"] = list(s["allowed_input_types"])

    return s


def load_options_from_yaml(settings_path: str | Path | None) -> Tuple[DoclingOptions, Dict[str, Any], Optional[Path]]:
    """
    Load DoclingOptions from a YAML file. Returns (options, raw_yaml_dict, resolved_path).
    If settings_path is None or file is missing, returns DoclingOptions() and {}.
    """
    if not settings_path:
        return DoclingOptions(), {}, None

    p = Path(settings_path).expanduser().resolve()
    if not p.exists():
        return DoclingOptions(), {}, p

    mgr = SettingsManager.instance()
    data, _ = mgr.get(str(p))
    norm = _apply_aliases(data)
    opts = DoclingOptions(**norm)
    return opts, data, p


# ------------------------------
# Public API used by services/pipeline
# ------------------------------
def get_settings(hint: Optional[str] = None) -> Dict[str, Any]:
    """Return the latest (hot-reloaded) YAML dict. If no file, returns {}."""
    data, _ = SettingsManager.instance().get(hint)
    return data


def apply_settings_overrides(
    options_obj: DoclingOptions,
    settings_path: str | Path | None = None,
    settings_dict: Dict[str, Any] | None = None,
) -> DoclingOptions:
    """
    Backward-compatible overlay:
      - Updates the given DoclingOptions *in place*, and also returns it.
      - If settings_path is provided, file values are applied first (hot-reloaded).
      - If settings_dict is provided, it takes precedence over file values.
      - Unknown fields are ignored by DoclingOptions.
      - Normalizes common aliases and fills sane defaults (e.g., num_threads=1).
    """
    if options_obj is None:
        options_obj = DoclingOptions()

    merged = options_obj.model_dump()

    # From file first
    if settings_path:
        file_opts, _, _ = load_options_from_yaml(settings_path)
        if isinstance(file_opts, DoclingOptions):
            merged.update({k: v for k, v in file_opts.model_dump().items() if v is not None})

    # Then from dict (wins)
    if settings_dict:
        normalized = _apply_aliases(settings_dict)
        merged.update({k: v for k, v in normalized.items() if v is not None})

    # Rehydrate and mutate in place so existing references keep the new values
    updated = DoclingOptions(**merged)
    for k, v in updated.model_dump().items():
        setattr(options_obj, k, v)

    return options_obj


def want_processors(settings_dict: Dict[str, Any]) -> Dict[str, bool]:
    """
    Convenience: compute which processors/exporters to attach from a raw settings dict.
    """
    def b(name: str, default: bool = False) -> bool:
        return bool(settings_dict.get(name, default))

    return {
        "header_footer": b("enable_header_footer_detection"),
        "label_correction": b("enable_label_correction", True),
        "export_markdown": b("export_markdown", True),
        "export_doc_json": b("export_doc_json", True),
        "export_deep_json_report": b("export_deep_json_report", True),
        "export_domtree_json": b("export_domtree_json", False),
    }
