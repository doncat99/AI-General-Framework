from __future__ import annotations

from typing import Iterator, Dict, Any
import json

from gateway.mem0.ingest.models import MemoryRecord
from workshop.assistant.ingest.sources.base import BaseSource
from workshop.assistant.ingest.utils import chunk_text
from workshop.assistant.setting import settings

class JSONRoadmapSource(BaseSource):
    def __init__(self, path: str) -> None:
        self.path = path

    def iter(self) -> Iterator[MemoryRecord]:
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)

        roadmap_title = data.get("roadmap_title", "Unknown Roadmap")
        source_url = data.get("source_url", "")
        sections = data.get("sections", [])

        for section in sections:
            sid = section.get("section_id")
            s_title = section.get("section_title")
            s_order = section.get("order")

            for topic in section.get("topics", []):
                tid = topic.get("topic_id")
                title = topic.get("topic_title", "")
                desc = topic.get("description", "")
                order = topic.get("order", -1)
                options = topic.get("options", [])

                base_text = f"{title}: {desc}".strip()
                if not base_text or base_text == ":":
                    continue

                meta: Dict[str, Any] = {
                    "roadmap_title": roadmap_title,
                    "source_url": source_url,
                    "section_id": sid,
                    "section_title": s_title,
                    "section_order": s_order,
                    "topic_id": tid,
                    "topic_title": title,
                    "topic_order": order,
                    "options": options,
                    "source": self.path,
                    "kind": "topic",
                }

                for idx, chunk in enumerate(chunk_text(base_text)):
                    m = {**meta, "chunk_index": idx}
                    yield MemoryRecord.from_text(
                        text=chunk,
                        metadata=m,
                        user_id=settings.MEM0_DEFAULT_USER_ID,
                        team_id=settings.MEM0_DEFAULT_TEAM_ID,
                        visibility=settings.MEM0_VISIBILITY,
                    )
