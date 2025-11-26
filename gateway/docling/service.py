# gateway/docling/service.py
from __future__ import annotations

import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from loguru import logger

from docling.datamodel.document import ConversionResult
from docling_core.types.doc import (
    DoclingDocument,
    DocItemLabel,
    BoundingBox,
    ImageRefMode,
    GroupLabel,
)
from docling_core.types.doc.document import (
    DocItem,
    GroupItem,
    ListItem,
    TableItem,
    PictureItem,
    SectionHeaderItem,
    TextItem,
    ContentLayer,
    KeyValueItem,
    FormItem,
    DescriptionAnnotation,
)

# Public docling-facing types / helpers re-exported by gateway.docling.__init__
from . import (
    DoclingOptions,
    get_converter,
    Artifact,
    Result,
    ResultStatus,
    BatchManifest,
)
# Settings helpers (hot-reload + overrides)
from .settings import apply_settings_overrides, get_settings

# ---------------------------------------------------------------------
# Helpers & Strict Mode
# ---------------------------------------------------------------------
DEFAULT_SETTINGS_BASENAME = "docling_settings.yaml"
ENV_SETTINGS_PATH = "DOCLING_SETTINGS"

class DoclingReadError(RuntimeError):
    """Raised when text extraction from exporter JSON fails in strict mode."""

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _locate_settings_file(base_dir_like: Optional[str | Path]) -> Optional[Path]:
    """
    Resolve docling_settings.yaml with a simple, predictable search order:
      1) DOCLING_SETTINGS env var (explicit path)
      2) next to this module (…/service.py -> …/docling_settings.yaml)
      3) under the provided base_dir_like
      4) CWD
    """
    # 1) Explicit env var
    env = os.getenv(ENV_SETTINGS_PATH)
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return p

    # 2) Typical places
    base_dir = Path(base_dir_like) if base_dir_like else None
    candidates = [
        (base_dir / DEFAULT_SETTINGS_BASENAME) if base_dir else None,
        Path(__file__).with_name(DEFAULT_SETTINGS_BASENAME),
        Path.cwd() / DEFAULT_SETTINGS_BASENAME,
    ]
    for cand in candidates:
        if cand and cand.is_file():
            return cand
    return None


def _effective_num_threads(opts: DoclingOptions) -> int:
    """
    Compute an effective worker count:
    - base it on hardware CPU count
    - clamp by opts.num_threads if provided
    - never less than 1
    """
    hw = max(1, (os.cpu_count() or 1))
    configured = getattr(opts, "num_threads", None)
    if configured is None:
        return hw
    try:
        configured_int = int(configured)
    except Exception:
        configured_int = hw
    return max(1, min(hw, configured_int))


def _name_endswith(p: Path, suffix: str) -> bool:
    return p.name.lower().endswith(suffix.lower())


def _resolve_artifacts_folder(options: DoclingOptions) -> str:
    """
    Prefer options.artifacts_folder_name, then legacy options.domtree_artifacts_folder.
    As a last resort, derive the default from MarkdownExporter().
    """
    for attr in ("artifacts_folder_name", "domtree_artifacts_folder"):
        v = getattr(options, attr, None)
        if isinstance(v, str) and v.strip():
            return v.strip()

    # Derive from plugin default (no hardcode).
    try:
        # Instantiating without args gives us the class' default.
        from .plugins import MarkdownExporter as _ME
        # Try both attributes to be tolerant of plugin versions.
        me = _ME()
        return getattr(me, "artifacts_folder_name", getattr(me, "domtree_artifacts_folder", "artifacts"))
    except Exception:
        # Absolute last resort; should rarely be used.
        return "artifacts"


def _save_artifact(
    base_dir: Path,
    artifacts_folder: str,
    input_stem: str,
    art: Artifact,
) -> Artifact:
    """
    Persist artifact.payload to disk if provided and path not already set.
    Files are saved under base_dir / <artifacts_folder> / <name>.
    Returns an Artifact with .path populated.
    """
    if art is None:
        return art
    if getattr(art, "path", None) is not None:
        return art
    if getattr(art, "payload", None) is None:
        return art

    target_dir = base_dir / artifacts_folder
    _ensure_dir(target_dir)

    # best-effort unique name fallback
    name = getattr(art, "name", None) or f"{input_stem}.bin"
    target_path = target_dir / name

    # avoid accidental overwrite if different payloads collide on name
    i = 1
    while target_path.exists():
        # if identical bytes, reuse
        try:
            if target_path.read_bytes() == art.payload:
                break
        except Exception:
            pass
        base = getattr(art, "name", None) or f"{input_stem}.bin"
        name = f"{base}.{i}"
        target_path = target_dir / name
        i += 1

    target_path.write_bytes(art.payload)
    art.path = target_path
    return art


def _safe_run_processor(
    base_dir: Path,
    conv: ConversionResult,
    processor: object,
) -> Tuple[List[Artifact], Dict[str, Any] | None]:
    """
    Run a single processor/exporter with strong exception isolation.
    Returns (artifacts, error_info). error_info is None if ok, otherwise a dict with details.
    """
    try:
        out = None
        if hasattr(processor, "process"):
            out = processor.process(str(base_dir), conv)
        elif callable(processor):
            out = processor(str(base_dir), conv)

        if out is None:
            return [], None
        if isinstance(out, list):
            return [a for a in out if a is not None], None
        return [out], None
    except Exception as e:
        err = {
            "processor": getattr(processor, "name", processor.__class__.__name__),
            "type": e.__class__.__name__,
            "message": repr(e),
            "traceback": traceback.format_exc(),
        }
        return [], err

# ---------------------------------------------------------------------
# DoclingService
# ---------------------------------------------------------------------
@dataclass
class DoclingService:
    """
    Shared service for docling conversion + processors/exporters + unified document helpers.
    """
    base_dir: str
    setting_dir: str | None = None
    options: DoclingOptions = field(default_factory=DoclingOptions)

    # internal caches
    _converted_for: set[str] = field(default_factory=set, init=False)
    _results: Dict[str, Result] = field(default_factory=dict, init=False)
    _artifact_root: Dict[str, Optional[Path]] = field(default_factory=dict, init=False)
    _doc_json_path: Dict[str, Optional[Path]] = field(default_factory=dict, init=False)
    _markdown_path: Dict[str, Optional[Path]] = field(default_factory=dict, init=False)
    _chapter_slice_registry: Dict[str, Tuple[int, int]] = field(default_factory=dict, init=False)

    def __post_init__(self):
        self._base_path = Path(self.base_dir)
        _ensure_dir(self._base_path)

    # -----------------------
    # Disk hydration (prevents needless reprocessing)
    # -----------------------
    def _find_existing_artifacts_on_disk(self, path: str) -> tuple[Optional[Path], Optional[Path], Optional[Path]]:
        """
        Scan self._base_path for prior artifacts of the given input.
        Returns: (artifact_root, doc_json_path, markdown_path)
        Prefers *.doc.json over generic *.json; ignores *deep_report.json and *domtree.json.
        """
        stem = Path(path).stem.lower()
        artifact_root: Optional[Path] = None
        doc_json: Optional[Path] = None
        md: Optional[Path] = None

        try:
            for p in self._base_path.rglob("*"):
                if not p.is_file():
                    continue
                # quick stem filter across the relative path
                hay = "/".join(part.lower() for part in p.parts)
                if stem not in hay:
                    continue

                # detect markdown
                if p.suffix.lower() in {".md", ".markdown"}:
                    if md is None:
                        md = p
                        artifact_root = artifact_root or p.parent
                    continue

                # detect doc json (prefer *.doc.json)
                if p.suffix.lower() == ".json":
                    nm = p.name.lower()
                    if nm.endswith(".doc.json"):
                        doc_json = p
                        artifact_root = artifact_root or p.parent
                        continue
                    if not (nm.endswith("deep_report.json") or nm.endswith("domtree.json")):
                        if doc_json is None:
                            doc_json = p
                            artifact_root = artifact_root or p.parent
        except Exception as e:
            logger.debug(f"[docling] disk scan failed while hydrating: {e}")

        return artifact_root, doc_json, md

    def _cache_artifacts_from_disk(self, path: str) -> None:
        root, doc_json, md = self._find_existing_artifacts_on_disk(path)
        self._artifact_root[path] = root
        self._doc_json_path[path] = doc_json
        self._markdown_path[path] = md
        self._converted_for.add(path)
        logger.debug(
            f"[docling] indexed_artifacts: file='{Path(path).name}', root='{root}', "
            f"doc_json={doc_json}, markdown={md}"
        )

    def _hydrate_from_disk(self, path: str) -> bool:
        """
        Populate internal caches from already-generated artifacts, if present.
        """
        root, doc_json, md = self._find_existing_artifacts_on_disk(path)
        if not (doc_json or md):
            return False

        # choose a reasonable root even if only one is present
        root = root or (doc_json.parent if doc_json else md.parent if md else self._base_path / "artifacts")

        self._artifact_root[path] = root
        self._doc_json_path[path] = doc_json
        self._markdown_path[path] = md
        self._converted_for.add(path)
        logger.debug(
            f"[docling] hydrated_from_disk: file='{Path(path).name}', "
            f"root='{root}', doc_json={doc_json}, markdown={md}"
        )
        return True

    # -----------------------
    # Conversion entry points
    # -----------------------
    def convert_path(self, path: str) -> Result:
        """
        Convert & post-process a single file path.
        Returns a Result with persisted artifacts.
        """
        in_path = Path(path)
        input_name = in_path.name
        input_stem = in_path.stem

        # Hot-load settings on every call
        try:
            settings_path = _locate_settings_file(self.setting_dir)
            settings_dict = get_settings(str(settings_path) if settings_path else None)
            if settings_dict:
                self.options = apply_settings_overrides(self.options, settings_dict=settings_dict)
                logger.info(f"Applied docling settings from: {settings_path}")
        except Exception as e:
            logger.error(f"Failed to load/overlay docling settings. error: {e}")
            # Non-fatal: continue with current options

        # Ensure options.num_threads respects hardware limits
        try:
            self.options.num_threads = _effective_num_threads(self.options)
        except Exception as e:
            logger.warning(f"Could not set effective num_threads; using as-is. err={e}")

        # Resolve artifacts folder (from options or plugin defaults)
        artifacts_folder = _resolve_artifacts_folder(self.options)
        _ensure_dir(self._base_path / artifacts_folder)

        logger.debug(
            f"[docling] convert_path start: file='{input_name}', "
            f"num_threads={getattr(self.options, 'num_threads', None)}, "
            f"artifacts_folder='{artifacts_folder}'"
        )

        # Convert
        try:
            converter = get_converter(self.options)
            conv: ConversionResult = converter.convert(in_path)
        except Exception as e:
            return Result(
                status=ResultStatus.FAILED,
                input_name=input_name,
                error=f"converter_failed: {e.__class__.__name__}: {repr(e)}\n{traceback.format_exc()}",
            )

        # Post-process/export
        processors = self._build_processors_for_options(artifacts_folder)
        artifacts: List[Artifact] = []
        post_errors: List[Dict[str, Any]] = []

        for proc in processors:
            out, err = _safe_run_processor(self._base_path, conv, proc)
            if out:
                for a in out:
                    a = _save_artifact(self._base_path, artifacts_folder, input_stem, a)
                    artifacts.append(a)
            if err:
                post_errors.append(err)

        if artifacts:
            logger.debug(
                "[docling] artifacts produced:\n" + "\n".join(
                    f" - {getattr(a, 'path', None)} ("
                    f"{(Path(getattr(a, 'path')).stat().st_size if getattr(a, 'path', None) and Path(getattr(a, 'path')).exists() else 0)} bytes)"
                    for a in artifacts
                )
            )
        if post_errors:
            logger.warning("[docling] exporter errors:\n" + "\n".join(str(e) for e in post_errors))

        meta = {
            "num_pages": len(conv.document.pages) if conv and conv.document else 0,
            "post_errors": post_errors if post_errors else [],
            "artifacts_folder": artifacts_folder,
            "num_threads": getattr(self.options, "num_threads", None),
        }

        res = Result(
            status=ResultStatus.SUCCESS,
            input_name=input_name,
            artifacts=artifacts,
            meta=meta,
        )

        # Cache artifact pointers for helpers (prefer exporter list; then reconcile with disk if needed)
        self._cache_artifacts_for_input(path, res)
        if (
            not self._doc_json_path.get(path)
            or (self._doc_json_path[path] and not self._doc_json_path[path].exists())
            or (self._markdown_path[path] and not self._markdown_path[path].exists())
        ):
            # Fall back to a disk scan to be resilient to plugin/path variations
            self._cache_artifacts_from_disk(path)

        return res

    def _build_processors_for_options(self, artifacts_folder: str) -> List[object]:
        """
        Build exporter list from options (tolerant across docling versions).
        """
        procs: List[object] = []
        try:
            from .plugins import MarkdownExporter, JsonExporter, DomTreeExporter, DeepJsonReportExporter
        except Exception as e:
            logger.warning(f"[docling] Could not import exporters: {e}")
            return procs

        # helper to instantiate with tolerant kwargs
        def _safe_new(cls, **kwargs):
            try:
                return cls(**kwargs)
            except TypeError:
                try:
                    return cls()
                except Exception:
                    return None

        if getattr(self.options, "export_markdown", True):
            me = _safe_new(MarkdownExporter, domtree_artifacts_folder=artifacts_folder)
            if me:
                procs.append(me)

        if getattr(self.options, "export_doc_json", True):
            je = _safe_new(JsonExporter)
            if je:
                procs.append(je)

        if getattr(self.options, "export_domtree_json", True):
            de = _safe_new(
                DomTreeExporter,
                domtree_artifacts_folder=artifacts_folder,
                merge_headers=True,
                merge_distance=50.0,
                strict_text=False,
                text_width=-1,
            )
            if de:
                procs.append(de)

        if getattr(self.options, "export_deep_json_report", True):
            re = _safe_new(DeepJsonReportExporter, domtree_artifacts_folder=artifacts_folder)
            if re:
                procs.append(re)

        logger.debug(f"[docling] exporters enabled: {[getattr(p, 'name', p.__class__.__name__) for p in procs]}")
        return procs

    def convert_bytes(self, filename: str, content: bytes) -> Result:
        tmp_dir = self._base_path / ".tmp"
        _ensure_dir(tmp_dir)
        tmp_path = tmp_dir / filename
        tmp_path.write_bytes(content)
        try:
            res = self.convert_path(str(tmp_path))
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
        return res

    def convert_batch(self, paths: Iterable[str]) -> BatchManifest:
        """
        Serial batch processing. Returns a BatchManifest.
        """
        return BatchManifest(results=[self.convert_path(p) for p in paths])

    def convert_batch_parallel(
        self,
        paths: Sequence[str],
        *,
        max_workers: Optional[int] = None,
        chunk_size: Optional[int] = None,
    ) -> BatchManifest:
        """
        Parallel batch (multi-process). Each worker builds its own converter.
        """
        items = list(paths)
        if not items:
            return BatchManifest(results=[])

        # Default worker count = effective num_threads (hardware-aware)
        if max_workers is None:
            max_workers = _effective_num_threads(self.options)
        if max_workers < 1:
            max_workers = 1

        # Default chunking tuned to worker count
        if chunk_size is None:
            from math import ceil
            chunk_size = max(1, ceil(len(items) / max_workers))

        chunks = [items[i: i + chunk_size] for i in range(0, len(items), chunk_size)]

        results: List[Result] = []

        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futs = [
                ex.submit(_worker_chunk, chunk, self.base_dir, self.setting_dir, self.options.model_dump())
                for chunk in chunks
            ]
            for fut in as_completed(futs):
                try:
                    results.extend(fut.result())
                except Exception as e:
                    # represent a worker-level failure as FAILED results for its inputs (unknown names)
                    results.append(
                        Result(
                            status=ResultStatus.FAILED,
                            input_name="__worker__",
                            error=f"worker_failed: {e.__class__.__name__}: {repr(e)}\n{traceback.format_exc()}",
                        )
                    )
        return BatchManifest(results=results)

    # -----------------------
    # Unified document utilities
    # -----------------------
    def emit_docling_artifacts(self, path: str) -> None:
        """
        Convert the input once and cache exporter paths.
        """
        self._ensure_converted(path)

    def get_doc_page_count(self, path: str) -> int:
        """
        Use the exporter DoclingDocument JSON to infer page count (len(page_infos or pages)); fallback to 1.
        (Kept non-strict to avoid breaking callers that only need a count.)
        """
        self._ensure_converted(path)
        doc_json = self._doc_json_path.get(path)
        if doc_json and doc_json.exists():
            try:
                d = DoclingDocument.load_from_json(doc_json)
                n = d.num_pages()
                # if n == 0:
                #     max_page = 0
                #     for item, _ in d.iterate_items():
                #         for prov in item.prov:
                #             if prov.page_no > max_page:
                #                 max_page = prov.page_no
                #     n = max(1, max_page)
                logger.debug(f"[docling] page_count: {n} ({doc_json})")
                return n
            except Exception as e:
                logger.debug(f"[docling] page_count failed to load {doc_json}: {e}")
        logger.debug("[docling] page_count fallback=1 (no doc_json)")
        return 1

    def read_doc_to_string(
        self,
        path: str,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
    ) -> str:
        """
        STRICT text assembly from exporter JSON.
          - If start_page/end_page provided -> return range; no markdown fallback; error if empty.
          - If omitted -> return whole doc from JSON; no markdown fallback; error if empty.
        Preference: pages[*].text -> page_infos[*].text -> paragraphs[*].text (prov-aware for ranges).
        """
        self._ensure_converted(path)
        doc_json = self._doc_json_path.get(path)
        md = self._markdown_path.get(path)

        range_requested = (start_page is not None) and (end_page is not None and end_page >= start_page)
        s = max(1, int(start_page or 1)) if range_requested else None
        e = int(end_page) if (range_requested and end_page and end_page >= s) else (s if range_requested else None)

        logger.debug(
            f"[docling][read] file='{Path(path).name}', "
            f"range_requested={range_requested} "
            f"{f'[{s}-{e}]' if range_requested else ''}, "
            f"has_doc_json={bool(doc_json)} has_markdown={bool(md)}"
        )

        if not (doc_json and doc_json.exists()):
            raise DoclingReadError(
                f"No exporter JSON (.doc.json) present for '{Path(path).name}'. "
                f"Cannot read {'range' if range_requested else 'document'} text strictly. "
                f"(markdown exists={bool(md)})"
            )

        if not range_requested:
            txt = self._read_text_from_doc_json_all(doc_json)
            logger.debug(f"[docling][read] strategy=all_from_doc_json chars={len(txt)}")
            if txt:
                return txt
            # Strict: error out
            raise DoclingReadError(
                f"Empty text for ALL-PAGES read from '{doc_json.name}'. "
                f"Check your exporters: export_domtree_json, JsonExporter schema, or OCR settings."
            )
        else:
            txt = self._read_text_from_doc_json_by_pages(doc_json, s, e)
            logger.debug(f"[docling][read] strategy=range_from_doc_json chars={len(txt)}")
            if txt:
                return txt
            # Strict: error out with diagnostics
            raise DoclingReadError(
                f"Empty text for RANGE [{s}-{e}] from '{doc_json.name}'. "
                f"No usable content in iterate_items with prov.page_no in range. "
                f"Verify: DomTree export enabled, item provenance populated, or adjust converter pipeline."
            )

    # --- Convenience helpers mirroring old handler API (kept here for pipelines) ---
    def get_range_text(self, path: str, start_page: int, end_page: int) -> Tuple[str, None]:
        text = self.read_doc_to_string(path, start_page, end_page)
        return text, None

    # -----------------------
    # Chapter-scoped view & artifact exports
    # -----------------------
    def emit_docling_artifacts_for_range(
        self,
        doc_path: str,
        chapter_folder: str,
        start_page: int,
        end_page: int,
        cid: str,
        artifacts_subdir: str = "artifacts",
    ) -> Tuple[Optional[Path], int]:
        """
        Copy full exporter JSON to <chapter_folder>/<cid>.doc.json and record absolute page range.
        """
        self._ensure_converted(doc_path)

        chapter_dir = Path(chapter_folder)
        chapter_dir.mkdir(parents=True, exist_ok=True)

        chapter_doc_json = chapter_dir / f"{cid}.doc.json"
        if chapter_doc_json.exists():
            # Already created; use recorded range if present
            abs_range = self._chapter_slice_registry.get(str(chapter_doc_json))
            if abs_range:
                s, e = abs_range
                logger.debug(f"[docling] chapter view exists {chapter_doc_json} range={s}-{e}")
                return chapter_doc_json, max(1, e - s + 1)
            # No recorded range? Assume the asked range
            self._chapter_slice_registry[str(chapter_doc_json)] = (start_page, end_page)
            return chapter_doc_json, max(1, end_page - start_page + 1)

        # Copy the full exporter JSON as the chapter JSON
        full_doc_json = self._doc_json_path.get(doc_path)
        if not full_doc_json or not full_doc_json.exists():
            logger.warning(f"[docling] no exporter document JSON for {doc_path}; cannot create chapter view.")
            return None, 0

        try:
            chapter_doc_json.write_bytes(full_doc_json.read_bytes())
            self._chapter_slice_registry[str(chapter_doc_json)] = (start_page, end_page)
            logger.debug(f"[docling] wrote chapter view {chapter_doc_json} range={start_page}-{end_page}")
        except Exception as e:
            logger.error(f"[docling] create chapter JSON failed for {cid}: {e}")
            return None, 0

        return chapter_doc_json, max(1, end_page - start_page + 1)

    def export_chapter_artifacts_from_doc(
        self,
        chapter_doc_json: Path,
        chapter_folder: str,
        slice_pages: Tuple[int, int],
        artifacts_subdir: str = "artifacts",
    ) -> None:
        """
        Export tables & pictures for the recorded absolute range of a chapter view.

        IMPORTANT:
        - `chapter_doc_json` is a COPY of the full document's .doc.json (not reindexed).
        - We look up the absolute (start_page, end_page) from the internal registry.
        - If not found, we fallback to treat `slice_pages` as absolute.
        """
        abs_range = self._chapter_slice_registry.get(str(chapter_doc_json)) or slice_pages
        abs_start, abs_end = abs_range

        chapter_dir = Path(chapter_folder)
        out_dir = chapter_dir / artifacts_subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        doc = self._load_docling_document_from_path(chapter_doc_json)
        if doc is None:
            logger.warning(f"[docling] No document JSON at {chapter_doc_json}; skipping artifacts.")
            return

        saved = 0

        # Tables
        for idx, table in enumerate(doc.tables):
            try:
                prov = table.prov[0] if table.prov else None
                page_no = prov.page_no if prov else None
                if page_no is None or not (abs_start <= page_no <= abs_end):
                    continue

                # CSV
                try:
                    df = table.export_to_dataframe(doc=doc)
                    (out_dir / f"table_{idx}.csv").write_text(df.to_csv(index=False))
                except Exception as e:
                    logger.debug(f"[docling] table dataframe export failed (page {page_no}, idx {idx}): {e}")

                # PNG
                try:
                    img = table.get_image(doc, prov_index=0)
                    if img:
                        with (out_dir / f"table_{idx}.png").open("wb") as fp:
                            img.save(fp, "PNG")
                except Exception as e:
                    logger.debug(f"[docling] table image export failed (page {page_no}, idx {idx}): {e}")

                saved += 1
            except Exception as e:
                logger.debug(f"[docling] skipping table {idx}: {e}")

        # Pictures
        for idx, pic in enumerate(doc.pictures):
            try:
                prov = pic.prov[0] if pic.prov else None
                page_no = prov.page_no if prov else None
                if page_no is None or not (abs_start <= page_no <= abs_end):
                    continue

                try:
                    img = pic.get_image(doc, prov_index=0)
                    if img:
                        with (out_dir / f"picture_{idx}.png").open("wb") as fp:
                            img.save(fp, "PNG")
                        saved += 1
                except Exception as e:
                    logger.debug(f"[docling] picture image export failed (page {page_no}, idx {idx}): {e}")
            except Exception as e:
                logger.debug(f"[docling] skipping picture {idx}: {e}")

        if saved == 0:
            logger.info(f"[docling] no tables/pictures in abs pages {abs_start}-{abs_end} for {chapter_doc_json}.")

    # -----------------------
    # Internals
    # -----------------------
    def _cache_artifacts_for_input(self, path: str, res: Result) -> None:
        """Populate internal caches from a new conversion result."""
        self._results[path] = res

        artifacts_folder = (res.meta or {}).get("artifacts_folder") or "artifacts"
        root = Path(self.base_dir) / artifacts_folder
        self._artifact_root[path] = root

        # pick exporter JSON & also record markdown if present
        doc_json = self._select_document_json(res, path)
        self._doc_json_path[path] = doc_json
        self._markdown_path[path] = self._select_markdown(res)
        self._converted_for.add(path)

        logger.debug(f"[docling] selected doc_json={doc_json}, markdown={self._markdown_path[path]}")

    def _ensure_converted(self, path: str) -> None:
        """
        Run Docling conversion once and cache exporter paths.
        First try to hydrate from existing artifacts on disk to avoid re-processing.
        """
        if path in self._converted_for:
            return

        # 1) Try hydrate from disk and return if successful
        if self._hydrate_from_disk(path):
            return

        # 2) Fall back to actual conversion
        res = self.convert_path(path)
        if not res or res.status != ResultStatus.SUCCESS:
            logger.error(f"[docling] conversion failed for {path}: {getattr(res, 'error', 'unknown')}")
            self._artifact_root[path] = None
            self._doc_json_path[path] = None
            self._markdown_path[path] = None
            self._converted_for.add(path)
            return
        # convert_path already cached (and reconciled with disk)

    @staticmethod
    def _select_markdown(res: Result) -> Optional[Path]:
        for art in getattr(res, "artifacts", []) or []:
            p = getattr(art, "path", None)
            if not p:
                continue
            p = Path(p)
            if p.suffix.lower() in {".md", ".markdown"}:
                return p
        return None

    @staticmethod
    def _select_document_json(res: Result, input_path: str) -> Optional[Path]:
        """
        Choose the exporter document JSON.
        Order:
          1) Any *.doc.json
          2) Any other *.json that successfully loads into DoclingDocument (skip deep_report/domtree)
        """
        from docling_core.types.doc import DoclingDocument as _Doc

        def _is_doc_json(p: Path) -> bool:
            return p.name.lower().endswith(".doc.json")

        def _is_excludable(p: Path) -> bool:
            n = p.name.lower()
            return n.endswith("deep_report.json") or n.endswith("domtree.json")

        artifacts = getattr(res, "artifacts", []) or []
        json_paths = []
        for a in artifacts:
            p = getattr(a, "path", None)
            if not p:
                continue
            q = Path(p)
            if q.suffix.lower() == ".json":
                json_paths.append(q)

        # 1) exact .doc.json
        for p in json_paths:
            if _is_doc_json(p):
                return p

        # 2) try any JSON that loads, skipping deep_report/domtree
        for p in json_paths:
            if _is_excludable(p):
                continue
            try:
                DoclingDocument.load_from_json(p)
                return p
            except Exception:
                continue

        logger.warning("[docling] No exporter document JSON found; some features may be unavailable.")
        return None

    @staticmethod
    def _load_docling_document_from_path(path: Path):
        try:
            return DoclingDocument.load_from_json(path)
        except Exception as e:
            logger.warning(f"[docling] Could not materialize DoclingDocument from {path}: {e}")
            return None

    # ---------- text assembly (ALL pages) ----------
    @staticmethod
    def _read_text_from_doc_json_all(doc_json_path: Path) -> str:
        """
        Assemble ALL text from a Docling `.doc.json` with robust fallbacks (within JSON only).

        Preference order:
          1) pages[*].text
          2) page_infos[*].text
          3) paragraphs[*].text (optionally sorted by provenance page_no if available)
        """
        d = DoclingDocument.load_from_json(doc_json_path)
        buf = []
        for item, _ in d.iterate_items():
            if isinstance(item, TextItem):
                buf.append(item.text)
            elif isinstance(item, TableItem):
                buf.append(item.export_to_markdown(doc=d))
            elif isinstance(item, PictureItem):
                if annos := item.get_annotations():
                    for anno in annos:
                        if isinstance(anno, DescriptionAnnotation):
                            buf.append(anno.text)
                buf.append(item.caption_text(d))
            elif isinstance(item, (KeyValueItem, FormItem)):
                buf.append(" ".join(cell.text for cell in item.graph.cells))
        return "\n\n".join(buf)
    # ---------- text assembly (RANGE) ----------
    @staticmethod
    def _read_text_from_doc_json_by_pages(doc_json_path: Path, start_page: int, end_page: int) -> str:
        """
        Defensive text assembly from exporter .doc.json for a page range [start_page, end_page].
        Preference order:
          1) pages[*].text
          2) page_infos[*].text
          3) paragraphs[*].text filtered by prov.page_no
        """
        d = DoclingDocument.load_from_json(doc_json_path)
        buf = []
        page_nrs = set(range(start_page, end_page + 1))
        for item, _ in d._iterate_items_with_stack(page_nrs=page_nrs):
            if isinstance(item, TextItem):
                buf.append(item.text)
            elif isinstance(item, TableItem):
                buf.append(item.export_to_markdown(doc=d))
            elif isinstance(item, PictureItem):
                if annos := item.get_annotations():
                    for anno in annos:
                        if isinstance(anno, DescriptionAnnotation):
                            buf.append(anno.text)
                buf.append(item.caption_text(d))
            elif isinstance(item, (KeyValueItem, FormItem)):
                buf.append(" ".join(cell.text for cell in item.graph.cells))
        return "\n\n".join(buf)

def _worker_chunk(paths: Sequence[str], base_dir: str, setting_dir: str | None, options_dict: dict) -> List[Result]:
    svc = DoclingService(base_dir=base_dir, setting_dir=setting_dir, options=DoclingOptions(**options_dict))
    try:
        svc.options.num_threads = _effective_num_threads(svc.options)
    except Exception:
        pass

    out: List[Result] = []
    for p in paths:
        out.append(svc.convert_path(p))
    return out
