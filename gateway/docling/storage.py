from __future__ import annotations

from pathlib import Path

from .types import Artifact


class StorageBackend:
    def save(self, base_dir: str, artifact: Artifact) -> Artifact:
        raise NotImplementedError


class LocalStorage(StorageBackend):
    def __init__(self, artifacts_folder: str = "artifacts"):
        self.artifacts_folder = artifacts_folder

    def save(self, base_dir: str, artifact: Artifact) -> Artifact:
        if artifact.path is not None or artifact.payload is None:
            return artifact
        out_dir = Path(base_dir) / self.artifacts_folder
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / artifact.name
        # avoid clobber
        i = 1
        while target.exists():
            alt = target.with_name(f"{target.stem}.{i}{target.suffix}")
            if not alt.exists():
                target = alt
                break
            i += 1
        target.write_bytes(artifact.payload)
        artifact.path = target
        return artifact


class MemoryStorage(StorageBackend):
    def save(self, base_dir: str, artifact: Artifact) -> Artifact:
        # no-op; payload stays in memory
        return artifact
