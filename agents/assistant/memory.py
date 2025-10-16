# agents/assistant/memory.py
from __future__ import annotations
from typing import List, Any
from loguru import logger
from mem0 import AsyncMemory

class Memory:
    def __init__(self) -> None:
        self._mem = AsyncMemory()

    async def add_user_text(self, user_id: str, text: str, **meta: Any) -> None:
        try:
            await self._mem.add(text, user_id=user_id, metadata=meta or None)
        except Exception as e:
            logger.warning(f"mem0 add failed: {e}")

    async def add_agent_text(self, user_id: str, text: str, **meta: Any) -> None:
        try:
            await self._mem.add(f"Agent: {text}", user_id=user_id, metadata=meta or None)
        except Exception as e:
            logger.warning(f"mem0 add failed: {e}")

    async def search(self, user_id: str, query: str, limit: int = 5) -> List[str]:
        try:
            res = await self._mem.search(query=query, user_id=user_id, limit=limit)
            if not res or "results" not in res:
                return []
            return [r.get("text", "") for r in res["results"] if r.get("text")]
        except Exception as e:
            logger.warning(f"mem0 search failed: {e}")
            return []
