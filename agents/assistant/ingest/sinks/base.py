from __future__ import annotations
from typing import List
from abc import ABC, abstractmethod
from agents.assistant.ingest.models import MemoryRecord

class BaseSink(ABC):
    @abstractmethod
    def upsert(self, records: List[MemoryRecord]) -> None:
        ...
