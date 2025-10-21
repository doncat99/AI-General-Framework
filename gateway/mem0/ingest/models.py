# gateway/mem0/sinks/models.py
from __future__ import annotations

from typing import Dict, Any
import hashlib
import json

from pydantic import BaseModel, Field


class MemoryRecord(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    user_id: str = "user-default"
    team_id: str = "team-default"
    visibility: str = "private"

    @staticmethod
    def deterministic_id(text: str, metadata: Dict[str, Any]) -> str:
        payload = json.dumps({"t": text, "m": metadata}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_text(
        cls, text: str, metadata: Dict[str, Any], user_id: str, team_id: str, visibility: str
    ) -> "MemoryRecord":
        rid = cls.deterministic_id(text, metadata)
        return cls(id=rid, text=text, metadata=metadata, user_id=user_id, team_id=team_id, visibility=visibility)
