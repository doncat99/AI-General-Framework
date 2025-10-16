import hashlib
import os
import uuid
from typing import Dict, Tuple, Optional

import json
import portalocker
from loguru import logger

from config import config

# Directory and file name for the file registry
REGISTRY_FILE = os.path.join(config.DOCUMENT_REGISTRY_PATH, "file_registry.json")


# -----------------------------
# Registry I/O
# -----------------------------

def ensure_registry_directory_exists():
    """Ensure the directory for storing the registry file exists."""
    os.makedirs(os.path.dirname(REGISTRY_FILE), exist_ok=True)


def load_registry() -> Dict[str, Dict[str, str]]:
    """Load the file registry from a JSON file."""
    ensure_registry_directory_exists()
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE, "r") as f:
            # If registry is corrupted, fall back to empty (prefer DB later).
            try:
                return json.load(f)
            except Exception:
                logger.warning("Registry file appears corrupted; starting with an empty registry.")
    return {}


def save_registry(registry: Dict[str, Dict[str, str]]):
    """Save the file registry to a JSON file (exclusive lock while writing)."""
    ensure_registry_directory_exists()
    with open(REGISTRY_FILE, "w") as f:
        portalocker.lock(f, portalocker.LOCK_EX)  # Acquire exclusive lock
        json.dump(registry, f, indent=4)
        portalocker.unlock(f)  # Release the lock


# -----------------------------
# Hashing utilities
# -----------------------------

def calculate_md5(file_path: str, chunk_size: int = 1024 * 1024 * 10) -> str:
    """Calculate the MD5 hash of a file in chunks."""
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            md5.update(chunk)
    return md5.hexdigest()


def calculate_partial_md5(file_path: str, chunk_size: int = 1024 * 1024 * 10) -> Tuple[str, str]:
    """Calculate the MD5 hash of the first and last parts of the file."""
    md5_start = hashlib.md5()
    md5_end = hashlib.md5()
    file_size = os.path.getsize(file_path)

    with open(file_path, "rb") as f:
        # Read the first chunk
        md5_start.update(f.read(chunk_size))

        # Seek to the end and read the last chunk (or reuse full file for tiny files)
        if file_size > chunk_size:
            f.seek(-chunk_size, os.SEEK_END)
            md5_end.update(f.read(chunk_size))
        else:
            # For very small files, hash the whole content for "end"
            f.seek(0, os.SEEK_SET)
            md5_end.update(f.read())

    return md5_start.hexdigest(), md5_end.hexdigest()


def _registry_key_for_file(file_path: str, chunk_size: int = 1024 * 1024 * 10) -> str:
    """Helper: build the registry composite key from partial MD5s."""
    start, end = calculate_partial_md5(file_path, chunk_size)
    return f"{start}-{end}"


# -----------------------------
# Public API
# -----------------------------

def get_file_id(file_path: str, url: Optional[str] = None, chunk_size: int = 1024 * 1024 * 10) -> str:
    """
    Retrieve or assign a UUID-agnostic *doc_id* for the file based on its partial MD5s and
    store URL if provided. Returns the canonical *doc_id* (the composite 'startmd5-endmd5').

    NOTE: This returns the *registry key* (content-based ID), not the UUID. The UUID is kept
    inside the registry entry and can be used by other systems if needed.
    """
    registry = load_registry()
    key = _registry_key_for_file(file_path, chunk_size)

    # Reuse existing entry
    if key in registry:
        if url and "url" not in registry[key]:
            registry[key]["url"] = url
            save_registry(registry)
        return registry[key]["uuid"]

    # Not found -> create entry (with UUID sidecar)
    file_uuid = str(uuid.uuid4())
    entry: Dict[str, Optional[str]] = {"uuid": file_uuid}
    if url:
        entry["url"] = url
    registry[key] = entry
    save_registry(registry)

    logger.debug(f"New file registered: key={key} uuid={file_uuid}")
    return file_uuid


def add_file_metadata(
    doc_id: Optional[str] = None,
    *,
    file_path: Optional[str] = None,
    real_name: Optional[str] = None,
    url: Optional[str] = None,
) -> None:
    """
    Enrich an existing registry entry with metadata.

    You can identify the entry by either:
      - doc_id: the composite key 'startmd5-endmd5' (preferred in your pipeline), or
      - file_path: we will derive the doc_id from the file bytes.

    If file_path is provided, we also capture:
      - original_filename
      - source_path
      - size_bytes
      - mtime
    """
    if not doc_id and not file_path:
        raise ValueError("add_file_metadata requires at least one of: doc_id or file_path")

    registry = load_registry()

    # Resolve doc_id from file_path if needed
    if not doc_id and file_path:
        doc_id = _registry_key_for_file(file_path)

    entry = registry.get(doc_id)  # type: ignore[arg-type]
    if not entry:
        logger.warning("add_file_metadata: registry entry not found for key=%s (call get_file_id first).", doc_id)
        return

    if file_path and os.path.exists(file_path):
        try:
            stat = os.stat(file_path)
            entry["original_filename"] = os.path.basename(file_path)
            entry["source_path"] = os.path.abspath(file_path)
            entry["size_bytes"] = stat.st_size  # type: ignore[assignment]
            entry["mtime"] = stat.st_mtime      # type: ignore[assignment]
        except Exception as e:
            logger.warning("add_file_metadata: could not stat %s (%s)", file_path, e)

    if real_name:
        entry["real_name"] = real_name
    if url:
        entry["url"] = url

    registry[doc_id] = entry  # type: ignore[index]
    save_registry(registry)


def verify_files_are_same(file1_path: str, file2_path: str, chunk_size: int = 1024 * 1024 * 10) -> bool:
    """Verify if two large files are the same by comparing MD5 hash (partial + full)."""

    # First, compare the start and end parts of the files
    md5_start1, md5_end1 = calculate_partial_md5(file1_path, chunk_size)
    md5_start2, md5_end2 = calculate_partial_md5(file2_path, chunk_size)

    if md5_start1 != md5_start2 or md5_end1 != md5_end2:
        return False

    # If the start and end hashes match, proceed to full comparison
    return calculate_md5(file1_path, chunk_size) == calculate_md5(file2_path, chunk_size)
