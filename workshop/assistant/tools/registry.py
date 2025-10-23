# agents/assistant/tools/registry.py
from __future__ import annotations

from typing import Callable, Dict

from workshop.assistant.tools.base_tool import BaseToolPack
from workshop.assistant.tools.box.mem0_tools import Mem0Tools


# Factory type for a tool pack
ToolPackFactory = Callable[..., BaseToolPack]

def _make_mem0_pack(**kw) -> BaseToolPack:
    """
    Factory for Mem0 tool pack.

    Recognized kwargs (optional):
      - mem0_base_url: explicit base URL override (else env MEM0_SERVER_URL)
      - mem0_prefix:   name prefix override (else env TOOL_PREFIX_MEM0)
    """
    base_url = kw.get("mem0_base_url")
    prefix = kw.get("mem0_prefix")
    return Mem0Tools.from_env(base_url=base_url, prefix=prefix)


# Central registry of available tool packs by name.
# Add new packs here (e.g., "search": _make_search_pack).
PACK_REGISTRY: Dict[str, ToolPackFactory] = {
    "mem0": _make_mem0_pack,
}
