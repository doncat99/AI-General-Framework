from __future__ import annotations
from typing import Iterable, List, Optional
from loguru import logger
from agents.assistant.ingest.models import MemoryRecord
from agents.assistant.ingest.sinks.base import BaseSink
from agents.assistant.ingest.sources.base import BaseSource
from agents.assistant.ingest.config import settings

def _batch(iterable: Iterable[MemoryRecord], size: int) -> Iterable[List[MemoryRecord]]:
    buf: List[MemoryRecord] = []
    for item in iterable:
        buf.append(item)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf

class IngestPipeline:
    def __init__(self, source: BaseSource, sinks: List[BaseSink], batch_size: Optional[int] = None) -> None:
        self.source = source
        self.sinks = sinks
        self.batch_size = batch_size or settings.BATCH_SIZE

    def run(self) -> None:
        total = 0
        for batch in _batch(self.source.iter(), self.batch_size):
            for sink in self.sinks:
                sink.upsert(batch)
            total += len(batch)
        logger.info(f"ingest complete: {total} records → {len(self.sinks)} sink(s)")
