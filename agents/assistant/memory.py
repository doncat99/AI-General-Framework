# agents/assistant/memory.py
from __future__ import annotations

from typing import List, Any, Iterable, Union, Dict
import inspect
from loguru import logger
from mem0 import AsyncMemory


def _coerce_text(item: Union[str, Dict[str, Any]]) -> str:
    """
    Normalize various mem0 result shapes to a plain string.
    Supports:
      - raw strings
      - dicts with 'text' / 'memory' / 'content'
    """
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        return (
            (item.get("text") or item.get("memory") or item.get("content") or "")
            .strip()
        )
    return ""


class Memory:
    """
    Thin async wrapper over mem0.AsyncMemory used by the AssistantAgent.

    - `add_user_text`  : store user utterances
    - `add_agent_text` : store assistant replies
    - `search`         : retrieve top-k related memories (as plain strings)

    mem0 is assumed to be installed and importable.
    """

    def __init__(self, **mem0_kwargs: Any) -> None:
        self._mem = AsyncMemory(**mem0_kwargs)

    async def add_user_text(self, user_id: str, text: str, **meta: Any) -> None:
        """Store a user utterance for the given user_id."""
        try:
            await self._mem.add(text, user_id=user_id, metadata=(meta or None))
        except Exception as e:
            logger.warning(f"mem0 add (user) failed: {e}")

    async def add_agent_text(self, user_id: str, text: str, **meta: Any) -> None:
        """Store an assistant reply for the given user_id."""
        try:
            await self._mem.add(f"Assistant: {text}", user_id=user_id, metadata=(meta or None))
        except Exception as e:
            logger.warning(f"mem0 add (assistant) failed: {e}")

    async def search(self, user_id: str, query: str, limit: int = 5) -> List[str]:
        """
        Retrieve top-k memories relevant to `query` and normalize to a list of strings.

        Handles both possible parameter names (`limit` or `top_k`) and
        multiple response shapes (dict with "results"/"data"/"memories" or a raw list).
        """
        try:
            # Detect whether mem0.search expects 'limit' or 'top_k'
            params = inspect.signature(self._mem.search).parameters
            karg = {"limit": limit} if "limit" in params else {"top_k": limit} if "top_k" in params else {}
            res = await self._mem.search(query=query, user_id=user_id, **karg)

            # Normalize response into an iterable of entries
            if isinstance(res, dict):
                candidates: Iterable[Any] = res.get("results") or res.get("data") or res.get("memories") or []
            else:
                candidates = res or []

            out: List[str] = []
            for item in candidates:
                t = _coerce_text(item)
                if t:
                    out.append(t)
            return out
        except Exception as e:
            logger.warning(f"mem0 search failed: {e}")
            return []
