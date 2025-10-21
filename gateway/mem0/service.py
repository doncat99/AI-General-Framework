#!/usr/bin/env python3
"""
Mem0 API Server for RAG Framework Integration (async, lifespan-ready)
- NeuraMemory(AsyncMemory) from memory.py
- Chroma retriever (Ollama embeddings)
- Sinks: chroma, mem0, neo4j
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import logging
import uvicorn
import inspect
import os

from .memory import NeuraMemory
from .config import build_mem0_config, sanitized_config
from .retrievers.chroma_retriever import ServerChromaRetriever, get_ollama_embedder

# Ingest sinks
from .ingest.models import MemoryRecord
from .ingest.sinks.base import BaseSink
from .ingest.sinks.chroma_sink import ChromaSink
from .ingest.sinks.mem0_sink import Mem0Sink
from .ingest.sinks.neo4j_sink import Neo4jSink

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mem0-server")

API_VERSION = os.getenv("API_VERSION", "1.6.0")


# ---------------- Helpers ----------------
async def _maybe_await(obj):
    return await obj if inspect.isawaitable(obj) else obj


async def _init_mem0_instance() -> NeuraMemory:
    cfg = build_mem0_config()
    logger.info("Initializing Mem0 (async) with config: %s", sanitized_config(cfg))
    # NOTE: from_config is async
    inst = await NeuraMemory.from_config(cfg)
    logger.info("Mem0 initialized successfully")
    return inst


def _init_retriever() -> ServerChromaRetriever:
    # Use a separate collection + path for the standalone retriever to avoid
    # clashing with Mem0’s internal Chroma instance.
    retr_collection = os.getenv("CHROMA_RETR_COLLECTION", "rag_index")
    retr_dir = os.getenv("CHROMA_RETR_PATH", "/app/data/chroma_retriever_db")

    return ServerChromaRetriever(
        persist_dir=retr_dir,
        collection=retr_collection,
        embed_fn=get_ollama_embedder(),
        distance="cosine",
    )


def _require_memory(request: Request) -> NeuraMemory:
    mem = getattr(request.app.state, "memory", None)
    if mem is None:
        raise HTTPException(status_code=503, detail="Mem0 not initialized")
    return mem


def _require_retriever(request: Request) -> ServerChromaRetriever:
    ret = getattr(request.app.state, "retriever", None)
    if ret is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    return ret


# ---------------- FastAPI Lifespan ----------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.init_error: Optional[str] = None

    # startup: memory
    try:
        app.state.memory = await _init_mem0_instance()
    except Exception as e:
        logger.exception("Failed to initialize Mem0 on startup")
        app.state.memory = None
        app.state.init_error = f"{type(e).__name__}: {e}"

    # startup: retriever
    try:
        app.state.retriever = await run_in_threadpool(_init_retriever)
    except Exception as e:
        logger.exception("Failed to initialize Chroma retriever on startup")
        app.state.retriever = None
        if not app.state.init_error:
            app.state.init_error = f"Retriever: {type(e).__name__}: {e}"

    # startup: sinks
    app.state.sinks: Dict[str, BaseSink] = {}
    try:
        if app.state.retriever is not None:
            app.state.sinks["chroma"] = ChromaSink(app.state.retriever)
        if app.state.memory is not None:
            app.state.sinks["mem0"] = Mem0Sink(app.state.memory)
    except Exception as e:
        logger.exception("Failed to init sinks on startup")
        if not app.state.init_error:
            app.state.init_error = f"Sinks: {type(e).__name__}: {e}"

    # neo4j optional
    try:
        if os.getenv("NEO4J_PASSWORD"):
            app.state.sinks["neo4j"] = Neo4jSink()
    except Exception as e:
        logger.warning(f"Neo4j sink disabled: {e}")

    try:
        yield
    finally:
        # shutdown hooks (best-effort)
        mem = getattr(app.state, "memory", None)
        if mem is not None:
            aclose = getattr(mem, "aclose", None)
            if callable(aclose):
                try:
                    await aclose()
                except Exception as e:
                    logger.warning("Error during Mem0 aclose(): %s", e)


# ---------------- FastAPI App ----------------
app = FastAPI(
    title="Mem0 Memory Service",
    version=API_VERSION,
    lifespan=lifespan,
)

# -------------------- Models (Mem0) --------------------
class AddMemoryRequest(BaseModel):
    message: str
    user_id: Optional[str] = "default_user"
    agent_id: Optional[str] = "rag_agent"
    metadata: Optional[Dict[str, Any]] = None


class SearchMemoryRequest(BaseModel):
    query: str
    user_id: Optional[str] = "default_user"
    agent_id: Optional[str] = "rag_agent"
    limit: Optional[int] = 5


class UpdateMemoryRequest(BaseModel):
    memory_id: str
    data: str


# -------------------- Models (Retriever) --------------------
class AddTextsRequest(BaseModel):
    texts: List[str]
    metadatas: Optional[List[Dict[str, Any]]] = None
    ids: Optional[List[str]] = None
    batch_size: Optional[int] = 256


class SearchIndexRequest(BaseModel):
    query: str
    k: Optional[int] = 4
    where: Optional[Dict[str, Any]] = None


class DeleteIdsRequest(BaseModel):
    ids: List[str]


# -------------------- Sink Ingest --------------------
class UpsertRequest(BaseModel):
    records: List[MemoryRecord]
    sinks: Optional[List[str]] = None  # e.g., ["chroma","mem0","neo4j"]


async def _call_sink_upsert(sink: BaseSink, records: List[MemoryRecord]) -> int:
    """
    Handles both sync and async sink.upsert implementations.
    """
    result = sink.upsert(records)
    if inspect.isawaitable(result):
        return int(await result)
    return int(result)


# -------------------- Core Endpoints --------------------
@app.get("/health")
async def health_check(request: Request):
    state = "healthy" if getattr(request.app.state, "memory", None) is not None else "degraded"
    return {
        "status": state,
        "service": "mem0",
        "version": API_VERSION,
        "config": sanitized_config(),
        "retriever_ready": getattr(request.app.state, "retriever", None) is not None,
        "sinks": list(getattr(request.app.state, "sinks", {}).keys()),
        "init_error": getattr(request.app.state, "init_error", None),
    }


@app.get("/config")
async def show_config():
    return {"config": sanitized_config()}


@app.post("/reload")
async def reload_config(request: Request):
    try:
        request.app.state.init_error = None
        request.app.state.memory = await _init_mem0_instance()
        request.app.state.retriever = await run_in_threadpool(_init_retriever)

        # rebuild sinks after components change
        sinks: Dict[str, BaseSink] = {}
        if request.app.state.retriever is not None:
            sinks["chroma"] = ChromaSink(request.app.state.retriever)
        if request.app.state.memory is not None:
            sinks["mem0"] = Mem0Sink(request.app.state.memory)
        if os.getenv("NEO4J_PASSWORD"):
            try:
                sinks["neo4j"] = Neo4jSink()
            except Exception as e:
                logger.warning(f"Neo4j sink disabled on reload: {e}")
        request.app.state.sinks = sinks

        return {"status": "success", "message": "Mem0 + retriever + sinks reloaded", "sinks": list(sinks.keys())}
    except Exception as e:
        logger.exception("Reload failed")
        request.app.state.init_error = f"{type(e).__name__}: {e}"
        raise HTTPException(status_code=500, detail=f"Reload failed: {e}")


# -------------------- Mem0 Memory APIs --------------------
@app.post("/add")
async def add_memory(request: Request, body: AddMemoryRequest):
    mem = _require_memory(request)
    try:
        result = await mem.add(
            messages=body.message,
            user_id=body.user_id,
            agent_id=body.agent_id,
            metadata=body.metadata or {},
        )
        return {"status": "success", "memories_added": len(result), "details": result}
    except Exception as e:
        logger.exception("Error adding memory")
        raise HTTPException(status_code=500, detail=f"Error adding memory: {e}")


@app.post("/search")
async def search_memory(request: Request, body: SearchMemoryRequest):
    mem = _require_memory(request)
    try:
        results = await mem.search(
            query=body.query,
            user_id=body.user_id,
            agent_id=body.agent_id,
            limit=body.limit,
        )
        return {"status": "success", "memories": results, "count": len(results)}
    except Exception as e:
        logger.exception("Error searching memory")
        raise HTTPException(status_code=500, detail=f"Error searching memory: {e}")


@app.get("/memories")
async def get_memories(request: Request, user_id: str = "default_user", agent_id: str = "rag_agent"):
    mem = _require_memory(request)
    try:
        results = await mem.get_all(user_id=user_id, agent_id=agent_id)
        return {"status": "success", "memories": results, "count": len(results)}
    except Exception as e:
        logger.exception("Error getting memories")
        raise HTTPException(status_code=500, detail=f"Error getting memories: {e}")


@app.put("/update")
async def update_memory(request: Request, body: UpdateMemoryRequest):
    mem = _require_memory(request)
    try:
        result = await mem.update(memory_id=body.memory_id, data=body.data)
        return {"status": "success", "updated": result}
    except Exception as e:
        logger.exception("Error updating memory")
        raise HTTPException(status_code=500, detail=f"Error updating memory: {e}")


@app.delete("/delete/{memory_id}")
async def delete_memory(request: Request, memory_id: str):
    mem = _require_memory(request)
    try:
        await mem.delete(memory_id=memory_id)
        return {"status": "success", "deleted": memory_id}
    except Exception as e:
        logger.exception("Error deleting memory")
        raise HTTPException(status_code=500, detail=f"Error deleting memory: {e}")


# -------------------- Chroma Retriever APIs --------------------
@app.post("/index/add")
async def index_add(request: Request, body: AddTextsRequest):
    retr = _require_retriever(request)
    try:
        added = await run_in_threadpool(
            retr.add_texts,
            body.texts,
            body.metadatas,
            body.ids,
            body.batch_size or 256,
        )
        return {"status": "success", "added": added}
    except Exception as e:
        logger.exception("Index add failed")
        raise HTTPException(status_code=500, detail=f"Index add failed: {e}")


@app.post("/index/upsert")
async def index_upsert(request: Request, body: AddTextsRequest):
    retr = _require_retriever(request)
    try:
        upserted = await run_in_threadpool(
            retr.upsert_texts,
            body.texts,
            body.metadatas,
            body.ids,
            body.batch_size or 256,
        )
        return {"status": "success", "upserted": upserted}
    except Exception as e:
        logger.exception("Index upsert failed")
        raise HTTPException(status_code=500, detail=f"Index upsert failed: {e}")


@app.post("/index/search")
async def index_search(request: Request, body: SearchIndexRequest):
    retr = _require_retriever(request)
    try:
        docs = await run_in_threadpool(
            retr.similarity_search, body.query, body.k or 4, body.where
        )
        return {"status": "success", "documents": docs, "count": len(docs)}
    except Exception as e:
        logger.exception("Index search failed")
        raise HTTPException(status_code=500, detail=f"Index search failed: {e}")


@app.post("/index/search_with_scores")
async def index_search_with_scores(request: Request, body: SearchIndexRequest):
    retr = _require_retriever(request)
    try:
        hits = await run_in_threadpool(
            retr.similarity_search_with_scores, body.query, body.k or 4, body.where
        )
        return {"status": "success", "hits": hits, "count": len(hits)}
    except Exception as e:
        logger.exception("Index search_with_scores failed")
        raise HTTPException(status_code=500, detail=f"Index search_with_scores failed: {e}")


@app.delete("/index/delete")
async def index_delete(request: Request, body: DeleteIdsRequest):
    retr = _require_retriever(request)
    try:
        n = await run_in_threadpool(retr.delete, body.ids)
        return {"status": "success", "deleted": n}
    except Exception as e:
        logger.exception("Index delete failed")
        raise HTTPException(status_code=500, detail=f"Index delete failed: {e}")


@app.get("/index/count")
async def index_count(request: Request):
    retr = _require_retriever(request)
    try:
        n = await run_in_threadpool(retr.count)
        return {"status": "success", "count": n}
    except Exception as e:
        logger.exception("Index count failed")
        raise HTTPException(status_code=500, detail=f"Index count failed: {e}")


# -------------------- Sink APIs --------------------
@app.post("/sink/upsert")
async def sink_upsert(request: Request, body: UpsertRequest):
    """
    Upsert MemoryRecord items into one or more sinks: chroma, mem0, neo4j.
    """
    if not body.records:
        return {"status": "success", "results": {}, "total": 0}

    all_sinks: Dict[str, BaseSink] = getattr(request.app.state, "sinks", {})
    if not all_sinks:
        raise HTTPException(status_code=503, detail="No sinks available")

    target_names = body.sinks or list(all_sinks.keys())
    unknown = [n for n in target_names if n not in all_sinks]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown sinks: {unknown}")

    results: Dict[str, int] = {}
    total = 0
    for name in target_names:
        try:
            n = await _call_sink_upsert(all_sinks[name], body.records)
            results[name] = n
            total += n
        except Exception:
            logger.exception("sink '%s' upsert failed", name)
            results[name] = -1

    return {"status": "success", "results": results, "total": total}


if __name__ == "__main__":
    # Bind to 8001 by default (or honor MEM0_PORT if provided)
    port = int(os.getenv("MEM0_PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
