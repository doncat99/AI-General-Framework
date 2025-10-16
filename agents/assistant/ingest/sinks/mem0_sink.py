from __future__ import annotations
from typing import List
from loguru import logger
from agents.assistant.ingest.sinks.base import BaseSink
from agents.assistant.ingest.models import MemoryRecord
from agents.assistant.ingest.config import settings

try:
    from mem0 import MemoryClient  # type: ignore
except Exception:
    MemoryClient = None  # type: ignore

class Mem0Sink(BaseSink):
    def __init__(self) -> None:
        if MemoryClient is None:
            raise RuntimeError("mem0ai not installed. `pip install mem0ai`")
        self.client = MemoryClient(api_key=settings.MEM0_API_KEY, base_url=settings.MEM0_BASE_URL)

    def upsert(self, records: List[MemoryRecord]) -> None:
        if not records:
            return
        ok, fail = 0, 0
        for r in records:
            try:
                self.client.add(
                    text=r.text,
                    metadata=r.metadata,
                    user_id=r.user_id,
                    team_id=r.team_id,
                    visibility=r.visibility,
                    external_id=r.id,
                )
                ok += 1
            except Exception as e:
                logger.warning(f"mem0 upsert failed for {r.id}: {e}")
                fail += 1
        logger.info(f"mem0 sink: {ok} ok, {fail} failed")
