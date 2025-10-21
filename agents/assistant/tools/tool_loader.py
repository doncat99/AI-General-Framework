# agents/assistant/tools/tool_loader.py
from __future__ import annotations

from typing import Optional, Sequence, Union, Type
from loguru import logger

from agents.assistant.adk_tool_client import AdkToolClient
from agents.assistant.tools.base_tool import BaseToolPack
from agents.assistant.tools.box.mem0_tools import Mem0Tools


PackSpec = Union[str, BaseToolPack, Type[BaseToolPack]]


def _resolve_pack(spec: PackSpec) -> BaseToolPack:
    """Turn a pack spec into a concrete instance."""
    if isinstance(spec, BaseToolPack):
        return spec
    if isinstance(spec, type) and issubclass(spec, BaseToolPack):
        return spec()
    if isinstance(spec, str):
        key = spec.strip().lower()
        if key in {"mem0", "mem0_tools"}:
            return Mem0Tools()
        raise ValueError(f"Unknown tool pack name: {spec!r}")
    raise TypeError(f"Unsupported pack spec: {type(spec)}")


def register_tools(
    client: AdkToolClient,
    packs: Optional[Sequence[PackSpec]] = None,
) -> int:
    """
    Register one or more tool packs on the given client.

    - If `packs` is None, defaults to [Mem0Tools()]
    - Each pack must be a BaseToolPack instance, a subclass, or a known name (e.g. "mem0")
    """
    packs = packs or [Mem0Tools()]
    total = 0
    for spec in packs:
        pack = _resolve_pack(spec)
        try:
            n = pack.register(client)
            total += n
            logger.info(f"Registered {n} tools from {pack.__class__.__name__}.")
        except Exception as e:
            logger.exception(f"Failed registering pack {pack.__class__.__name__}: {e}")
    return total
