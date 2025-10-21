# gateway/mem0/memory.py
"""
Custom AsyncMemory subclass for the Mem0 service.
Keeps project-specific behavior isolated from the FastAPI service.
"""
from __future__ import annotations
from typing import Any, Dict

from mem0 import AsyncMemory


class NeuraMemory(AsyncMemory):
    """
    Project-specific async memory class.
    We DO NOT override the instance's __class__ (that triggers CPython guard).
    Instead, we just return the native AsyncMemory that Mem0 builds.

    Add lightweight behavior tweaks by overriding methods (e.g., add()).
    """

    @classmethod
    async def from_config(cls, cfg: Dict[str, Any]) -> AsyncMemory:
        """
        Build the underlying AsyncMemory from cfg. Return it directly.
        (Mem0 controls the concrete class; we don't force-cast it.)
        """
        return await super().from_config(cfg)

    async def add(self, *args, **kwargs):
        """
        Example: sanitize metadata keys to strings (harmless normalization).
        """
        md = kwargs.get("metadata")
        if isinstance(md, dict):
            kwargs["metadata"] = {str(k): v for k, v in md.items()}
        return await super().add(*args, **kwargs)
