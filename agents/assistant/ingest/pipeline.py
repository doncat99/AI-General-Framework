# agents/assistant/ingest/pipeline.py
from __future__ import annotations
from typing import Iterable, List, Optional, Sequence, Dict, Any
import os
import httpx
import time

from loguru import logger

from gateway.mem0.ingest.models import MemoryRecord
from agents.assistant.ingest.sources.base import BaseSource
from agents.assistant.setting import settings


# -------- helpers --------
def _batch(iterable: Iterable[MemoryRecord], size: int) -> Iterable[List[MemoryRecord]]:
    buf: List[MemoryRecord] = []
    for item in iterable:
        buf.append(item)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


def _rec_to_dict(r: MemoryRecord) -> Dict[str, Any]:
    # pydantic v1/v2 compat
    try:
        return r.model_dump()  # type: ignore[attr-defined]
    except Exception:
        return r.dict()  # type: ignore[attr-defined]


def _resolve_mem0_url() -> str:
    # priority: explicit settings -> ENV -> default localhost
    url = None
    if settings is not None:
        url = getattr(settings, "MEM0_SERVER_URL", None)
        if not url:  # sometimes people only define port
            port = getattr(settings, "MEM0_PORT", None)
            if port:
                url = f"http://localhost:{port}"
    if not url:
        url = os.getenv("MEM0_SERVER_URL")
    if not url:
        port = os.getenv("MEM0_PORT", "8001")
        url = f"http://localhost:{port}"
    return url.rstrip("/")


class IngestPipeline:
    """
    Pushes MemoryRecord batches to the service /sink/upsert endpoint.

    Args:
        source:       BaseSource yielding MemoryRecord items
        sink_names:   target sinks on the server (e.g., ["chroma","mem0","neo4j"]);
                      None => let server use all available sinks
        batch_size:   overrides default batch size; if omitted uses settings.BATCH_SIZE or 100
        timeout:      HTTP timeout in seconds
        max_retries:  simple retry count for transient HTTP failures
    """

    def __init__(
        self,
        source: BaseSource,
        sink_names: Optional[Sequence[str]] = None,
        batch_size: Optional[int] = None,
        timeout: float = 30.0,
        max_retries: int = 1,
    ) -> None:
        self.source = source
        self.sink_names = list(sink_names) if sink_names else None
        self.batch_size = (
            batch_size
            or (getattr(settings, "BATCH_SIZE", None) if settings is not None else None)
            or 100
        )
        self.base_url = _resolve_mem0_url()
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def _post_upsert(self, records: List[MemoryRecord]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"records": [_rec_to_dict(r) for r in records]}
        if self.sink_names:
            payload["sinks"] = self.sink_names

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.post("/sink/upsert", json=payload)
                resp.raise_for_status()
                return resp.json()
            except (httpx.HTTPError, httpx.TransportError) as e:
                last_exc = e
                if attempt < self.max_retries:
                    backoff = 0.5 * (2 ** attempt)
                    logger.warning(f"/sink/upsert failed (attempt {attempt+1}), retrying in {backoff:.1f}s: {e}")
                    time.sleep(backoff)
                else:
                    break
        assert last_exc is not None
        raise last_exc

    def run(self) -> None:
        # optional: ping health (non-fatal)
        try:
            r = self._client.get("/health")
            if r.status_code == 200:
                info = r.json()
                logger.info(
                    f"mem0_service up: version={info.get('version')} sinks={info.get('sinks')}"
                )
            else:
                logger.warning(f"mem0_service health: HTTP {r.status_code}")
        except Exception as e:  # not fatal
            logger.warning(f"mem0_service health check failed: {e}")

        total = 0
        batches = 0
        for batch in _batch(self.source.iter(), self.batch_size):
            batches += 1
            try:
                res = self._post_upsert(batch)
                results = res.get("results", {})
                total += len(batch)
                logger.info(
                    f"batch {batches}: upserted {len(batch)} → {results} (total={total})"
                )
            except Exception as e:
                logger.error(f"batch {batches}: upsert failed ({len(batch)} records): {e}")

        logger.info(
            f"ingest complete: {total} records → remote sinks {self.sink_names or 'ALL'} @ {self.base_url}"
        )

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass

    def __enter__(self) -> "IngestPipeline":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
