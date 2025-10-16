from __future__ import annotations
from typing import Iterator, AsyncIterator
from abc import ABC, abstractmethod
from agents.assistant.ingest.models import MemoryRecord

class BaseSource(ABC):
    @abstractmethod
    def iter(self) -> Iterator[MemoryRecord]:
        ...

    async def aiter(self) -> AsyncIterator[MemoryRecord]:
        for item in self.iter():
            yield item
