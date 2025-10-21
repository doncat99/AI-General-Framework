# gateway/mem0/sinks/base.py
from __future__ import annotations

from typing import List
from abc import ABC, abstractmethod

from ..models import MemoryRecord


class BaseSink(ABC):
    """
    Base interface for sinks.

    NOTE: Implementations MAY return an awaitable (async) or a plain int.
    The server will detect and await if needed.
    Return value should be the count of upserted records.
    """

    @abstractmethod
    def upsert(self, records: List[MemoryRecord]) -> int:
        ...
