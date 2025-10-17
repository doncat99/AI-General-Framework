# agents/assistant/memory.py
from __future__ import annotations

from typing import List, Any, Optional, Iterable, Tuple
from loguru import logger
from difflib import SequenceMatcher
from pathlib import Path
import json
import time
import os
import asyncio
import inspect

from .setting import settings

# ------------------ early env guards (run BEFORE SDK imports) ------------------

def _ensure_no_proxy_localhost() -> None:
    """
    Make sure localhost isn't routed through proxies. Otherwise Ollama at
    127.0.0.1:11434 can fail with httpx RemoteProtocolError via the proxy.
    """
    cur = os.getenv("NO_PROXY") or os.getenv("no_proxy") or ""
    want = {"localhost", "127.0.0.1", "::1"}
    have = {x.strip() for x in cur.split(",") if x.strip()}
    if not want.issubset(have):
        new = ",".join(sorted(have | want))
        os.environ["NO_PROXY"] = new
        os.environ["no_proxy"] = new

_ensure_no_proxy_localhost()

from mem0 import AsyncMemory  # noqa: E402  (import after env guards)


# ---------------------------- helpers ----------------------------

def _bool_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip().lower()
    return v in {"1", "true", "yes", "y", "on"}

def _dedupe_preserve(seq: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for s in seq:
        s_norm = (s or "").strip()
        if not s_norm or s_norm in seen:
            continue
        seen.add(s_norm)
        out.append(s_norm)
    return out


def _extract_text(item: Any) -> Optional[str]:
    """
    Extract human-readable memory text from many possible shapes returned by mem0.
    Intentionally forgiving; never raises.
    """
    if isinstance(item, str):
        return item

    if isinstance(item, dict):
        for k in ("memory", "text", "content", "data", "value"):
            v = item.get(k)
            if isinstance(v, str) and v.strip():
                return v
        for nest_key in ("payload", "metadata"):
            nested = item.get(nest_key)
            if isinstance(nested, dict):
                for k in ("memory", "text", "content", "data", "value"):
                    v = nested.get(k)
                    if isinstance(v, str) and v.strip():
                        return v
        res = item.get("result")
        if isinstance(res, dict):
            v = res.get("memory") or res.get("text")
            if isinstance(v, str) and v.strip():
                return v
        return None

    # attribute-style access (safe)
    for attr in ("memory", "text", "content"):
        try:
            v = getattr(item, attr, None)
            if isinstance(v, str) and v.strip():
                return v
        except Exception:
            pass

    # if an object exposes a dict-like "data"
    try:
        data = getattr(item, "data", None)
        if isinstance(data, dict):
            for k in ("memory", "text", "content", "value"):
                v = data.get(k)
                if isinstance(v, str) and v.strip():
                    return v
    except Exception:
        pass

    return None


def _normalize_results(payload: Any) -> List[str]:
    """
    Normalize mem0 search/get_all payloads to a list[str].
    """
    if isinstance(payload, dict):
        items = payload.get("results")
        if isinstance(items, list):
            return _dedupe_preserve(filter(None, (_extract_text(i) for i in items)))
        for alt in ("data", "items", "hits", "memories"):
            items = payload.get(alt)
            if isinstance(items, list):
                return _dedupe_preserve(filter(None, (_extract_text(i) for i in items)))
        single = _extract_text(payload)
        return _dedupe_preserve([single] if single else [])

    if isinstance(payload, list):
        return _dedupe_preserve(filter(None, (_extract_text(i) for i in payload)))

    for attr in ("results", "data", "items", "hits", "memories"):
        try:
            items = getattr(payload, attr, None)
            if isinstance(items, list):
                return _dedupe_preserve(filter(None, (_extract_text(i) for i in items)))
        except Exception:
            pass

    single = _extract_text(payload)
    return _dedupe_preserve([single] if single else [])


def _describe(obj: Any, *, maxlen: int = 120) -> str:
    try:
        t = type(obj).__name__
        if isinstance(obj, (list, tuple)):
            inner = type(obj[0]).__name__ if obj else "empty"
            extra = f"len={len(obj)}[{inner}]"
            return f"{t}({extra})"
        if isinstance(obj, dict):
            return f"dict(keys={list(obj.keys())[:5]})"
        s = repr(obj)
        if len(s) > maxlen:
            s = s[:maxlen] + "…"
        return f"{t}:{s}"
    except Exception:
        return f"<{type(obj).__name__}>"


# ----------------------- local fallback store -----------------------

def _local_path(user_id: str) -> Path:
    safe = "".join(c for c in (user_id or "default") if c.isalnum() or c in ("-", "_"))
    return os.path.join(settings.MEM_FALLBACK_DIR, f"{safe}.jsonl")


def _local_append(user_id: str, role: str, text: str) -> None:
    rec = {"ts": time.time(), "role": role, "text": text}
    p = _local_path(user_id)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _local_load_texts(user_id: str) -> List[str]:
    p = _local_path(user_id)
    if not p.exists():
        return []
    out: List[str] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                t = obj.get("text")
                if isinstance(t, str) and t.strip():
                    out.append(t)
            except Exception:
                continue
    return out


# --------------- config + construction helpers for mem0 ---------------

def _resolve_openai_base_url() -> Optional[str]:
    for key in ("MEM0_OPENAI_BASE_URL", "OPENAI_BASE_URL", "OPENAI_API_BASE", "OPENAI_API_HOST"):
        v = os.getenv(key)
        if v and v.strip():
            return v.strip()
    return None


def _resolve_ollama_base_url() -> str:
    return (os.getenv("MEM0_OLLAMA_BASE_URL")
            or os.getenv("OLLAMA_BASE_URL")
            or "http://127.0.0.1:11434").strip()


def _is_auth_error(exc: Exception) -> bool:
    s = str(exc).lower()
    return (
        ("401" in s)
        or ("invalid api key" in s)
        or ("expired" in s)
        or ("unauthorized" in s)
        or ("已过期" in s)  # token expired (zh)
    )

def _is_ollama_pull_error(exc: Exception) -> bool:
    s = str(exc).lower()
    return ("ollama" in s and ("pull" in s or "manifest" in s or "registry" in s)) or "EOF (status code: 500)" in s


def _resolve_provider() -> str:
    """
    STRICT provider resolution:
      - If MEM0_EMBEDDER or MEM0_EMBEDDER_PROVIDER is set, respect it (ollama|openai).
      - Otherwise DEFAULT TO 'ollama' to avoid surprising OpenAI calls.
    """
    env = (os.getenv("MEM0_EMBEDDER") or os.getenv("MEM0_EMBEDDER_PROVIDER") or "").strip().lower()
    if env in {"ollama", "openai"}:
        return env
    return "ollama"


def _default_dims_for(provider: str, model: str | None) -> int:
    if provider == "ollama":
        return int(os.getenv("MEM0_EMBEDDING_DIMS") or "768")
    # openai
    m = (model or "").strip()
    if m.startswith("text-embedding-3-large"):
        return int(os.getenv("MEM0_EMBEDDING_DIMS") or "3072")
    return int(os.getenv("MEM0_EMBEDDING_DIMS") or "1536")


def _compute_collection_name(base_prefix: str, provider: str, dims: int) -> str:
    suffix = f"_{provider}_{dims}"
    if base_prefix.endswith(suffix):
        return base_prefix
    return f"{base_prefix}{suffix}"


def _ensure_qdrant_collection(path: str, name: str, dims: int, recreate_on_mismatch: bool = False) -> None:
    """
    Ensure the local Qdrant collection exists with the expected dims.
    If `recreate_on_mismatch=True` and dims mismatch, recreate it.
    """
    try:
        from qdrant_client import QdrantClient, models  # type: ignore
    except Exception:
        logger.debug("qdrant_client not installed; skipping collection dimension check.")
        return

    client = QdrantClient(path=path, prefer_grpc=False)
    try:
        info = client.get_collection(name)
        try:
            current = info.config.params.vectors.size
        except Exception:
            current = getattr(getattr(info, "vectors_count", None), "size", None)
        if current != dims:
            msg = (
                f"Qdrant collection '{name}' has size={current}, expected={dims}. "
                f"{'Recreating' if recreate_on_mismatch else 'Leaving as-is'}."
            )
            if recreate_on_mismatch:
                logger.warning(msg)
                client.recreate_collection(
                    collection_name=name,
                    vectors_config=models.VectorParams(size=dims, distance=models.Distance.COSINE),
                )
            else:
                logger.warning(msg)
        else:
            logger.debug(f"Qdrant collection '{name}' dims OK ({dims}).")
    except Exception:
        logger.info(f"Qdrant collection '{name}' not found; creating fresh with dims={dims}.")
        from qdrant_client import models  # type: ignore
        client.recreate_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=dims, distance=models.Distance.COSINE),
        )


def _build_cfg(embedder_provider: str) -> tuple[dict, str, int, str, str, str, str]:
    """
    Build a mem0 config dict based on env + chosen embedder provider.
    Returns (cfg, provider, dims, collection_name, llm_provider, qdrant_path, embed_model)
    """
    provider = (embedder_provider or "ollama").lower().strip()
    if provider not in ("openai", "ollama"):
        provider = "ollama"

    # Vector store path and base name (from app settings)
    qdrant_path = settings.MEM0_QDRANT_PATH
    base_collection = settings.MEM0_QDRANT_COLLECTION

    # Embedder
    if provider == "ollama":
        embed_model = os.getenv("MEM0_EMBEDDING_MODEL", "nomic-embed-text").strip()
        dims = _default_dims_for(provider, embed_model)
        embedder_cfg = {
            "provider": "ollama",
            "config": {
                "ollama_base_url": _resolve_ollama_base_url(),
                "model": embed_model,
                "embedding_dims": dims,
            },
        }
    else:
        embed_model = os.getenv("MEM0_EMBEDDING_MODEL", "text-embedding-3-small").strip()
        dims = _default_dims_for(provider, embed_model)
        embedder_cfg = {
            "provider": "openai",
            "config": {
                "api_key": os.getenv("MEM0_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or "",
                **({"openai_base_url": _resolve_openai_base_url()} if _resolve_openai_base_url() else {}),
                "model": embed_model,
                "embedding_dims": dims,
            },
        }

    # LLM provider (default ollama to avoid accidental OpenAI usage)
    llm_provider = (os.getenv("MEM0_LLM") or "ollama").strip().lower()
    if llm_provider == "openai":
        llm_cfg = {
            "provider": "openai",
            "config": {
                "api_key": os.getenv("MEM0_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or "",
                **({"openai_base_url": _resolve_openai_base_url()} if _resolve_openai_base_url() else {}),
                **({"model": os.getenv("MEM0_LLM_MODEL")} if os.getenv("MEM0_LLM_MODEL") else {}),
            },
        }
    else:
        llm_cfg = {
            "provider": "ollama",
            "config": {
                "ollama_base_url": _resolve_ollama_base_url(),
                "model": os.getenv("MEM0_LLM_MODEL", "llama3.1"),
            },
        }

    # Vector store (Qdrant) with explicit dims
    collection = _compute_collection_name(base_collection, provider, dims)
    vs_cfg = {
        "provider": "qdrant",
        "config": {
            "collection_name": collection,
            "path": qdrant_path,
            "embedding_model_dims": dims,
            # NOTE: do NOT include "distance" here; Mem0's pydantic rejects extra fields
        },
    }

    # Ensure Qdrant collection dims match (optional auto-recreate)
    _ensure_qdrant_collection(
        qdrant_path, collection, dims,
        recreate_on_mismatch=_bool_env("MEM0_QDRANT_RECREATE_ON_MISMATCH", False),
    )

    cfg: dict = {"llm": llm_cfg, "embedder": embedder_cfg, "vector_store": vs_cfg}
    return cfg, provider, dims, collection, llm_provider, qdrant_path, embed_model


async def _from_config_async(cfg: dict) -> AsyncMemory:
    """Await AsyncMemory.from_config safely."""
    mem_ctor = getattr(AsyncMemory, "from_config", None)
    if not callable(mem_ctor):
        return AsyncMemory()
    obj = mem_ctor(cfg)
    if inspect.iscoroutine(obj):
        return await obj  # type: ignore[return-value]
    return obj  # type: ignore[return-value]


def _instantiate_from_config_sync(cfg: dict) -> tuple[Optional[AsyncMemory], bool]:
    """
    Try to build AsyncMemory.from_config synchronously when we are NOT inside a running loop.
    Returns (instance_or_None, needs_async_rebuild_flag).
    """
    mem_ctor = getattr(AsyncMemory, "from_config", None)
    if not callable(mem_ctor):
        return AsyncMemory(), False

    # If from_config is async and a loop is already running, don't create the coroutine.
    if inspect.iscoroutinefunction(mem_ctor):
        try:
            asyncio.get_running_loop()
            # loop is running -> we cannot await here; defer to async rebuild
            return None, True
        except RuntimeError:
            # no running loop; safe to await via asyncio.run
            return asyncio.run(mem_ctor(cfg)), False

    # from_config is sync; call it
    obj = mem_ctor(cfg)
    # Defensive: some libs return a coroutine even if the function isn't marked async
    if inspect.iscoroutine(obj):
        try:
            asyncio.get_running_loop()
            return None, True
        except RuntimeError:
            return asyncio.run(obj), False
    return obj, False


def _apply_env_for_fallback(provider: str, dims: int, embed_model: str, qdrant_path: str, collection: str, llm_provider: str) -> None:
    """
    If we must fall back to env-based AsyncMemory(), set env so it mirrors our explicit cfg.
    """
    os.environ["MEM0_EMBEDDER"] = provider
    os.environ["MEM0_EMBEDDER_PROVIDER"] = provider
    os.environ["MEM0_EMBEDDING_MODEL"] = embed_model
    os.environ["MEM0_EMBEDDING_DIMS"] = str(dims)
    os.environ["MEM0_QDRANT_PATH"] = qdrant_path
    os.environ["MEM0_QDRANT_COLLECTION"] = collection
    os.environ["MEM0_LLM"] = llm_provider


# ------------------------------ Memory ------------------------------

class Memory:
    """
    Robust adapter around mem0.AsyncMemory with explicit provider selection.
    - Defaults to **Ollama** (no surprise OpenAI usage).
    - If Ollama embedder can't be pulled/initialized, we DO NOT switch to OpenAI unless you opt-in.
      We continue in **local fallback** mode (lexical search, JSONL persistence).
    """

    def __init__(self) -> None:
        # Build explicit config
        cfg, prov, dims, col, llm_prov, qdrant_path, embed_model = _build_cfg(_resolve_provider())
        self._cfg = cfg
        self._embedder_provider = prov
        self._embed_dims = dims
        self._collection = col
        self._llm_provider = llm_prov
        self._qdrant_path = qdrant_path
        self._embed_model = embed_model

        # Fallback knobs
        self._ollama_fallback_enabled = _bool_env("MEM0_OLLAMA_FALLBACK", False)  # allow explicit switch to ollama if started on openai
        self._openai_fallback_enabled = _bool_env("MEM0_OPENAI_FALLBACK", False)  # allow explicit switch to openai if ollama fails

        # Deferred build flags
        self._mem_disabled: bool = False  # if true, skip mem0 calls and use local-only
        mem, needs_async = _instantiate_from_config_sync(cfg)
        if mem is None:
            # Avoid mem0 defaulting to OpenAI by mirroring our cfg into env for the interim AsyncMemory()
            _apply_env_for_fallback(prov, dims, embed_model, qdrant_path, col, llm_prov)
            mem = AsyncMemory()
        self._mem: AsyncMemory = mem
        self._needs_async_rebuild: bool = needs_async

        # Log effective providers and key config
        try:
            logger.info(
                f"mem0 cfg: embedder={self._embedder_provider} llm={self._llm_provider} "
                f"vector_store=qdrant endpoint={self._qdrant_path}"
            )
            if self._embedder_provider == "ollama":
                logger.debug(
                    f"mem0 ollama_base={_resolve_ollama_base_url()} "
                    f"embedder={self._embedder_provider} model={self._embed_model} dims={self._embed_dims}"
                )
            else:
                base_url = _resolve_openai_base_url() or "<default>"
                key_set = bool(os.getenv("MEM0_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"))
                logger.debug(
                    f"mem0 openai_base={base_url} key_set={key_set} "
                    f"embedder={self._embedder_provider} model={self._embed_model} dims={self._embed_dims}"
                )
        except Exception:
            pass

    # ---------- internal: diagnostics & rebuild ----------

    async def _ensure_async_rebuild(self) -> None:
        """If we had to defer from_config (loop was running), finalize it now. On failure, enter local-only mode."""
        if self._mem_disabled or not self._needs_async_rebuild:
            return
        try:
            self._mem = await _from_config_async(self._cfg)
            self._needs_async_rebuild = False
            logger.debug("[rebuild] finalized AsyncMemory.from_config in async context.")
        except Exception as e:
            logger.error(f"[rebuild] async from_config failed: {e}")
            # If Ollama pull/registry failed, don't try OpenAI unless explicitly allowed.
            if _is_ollama_pull_error(e) and self._embedder_provider == "ollama":
                if self._openai_fallback_enabled:
                    await self._switch_provider("openai")
                    return
                # Hard-disable mem0; run local-only
                self._mem_disabled = True
                logger.warning("[rebuild] Ollama unavailable; running in local-only (lexical) mode.")
            else:
                # Other errors: also disable to avoid surprise network calls (e.g., OpenAI via Langfuse wrappers)
                self._mem_disabled = True
                logger.warning("[rebuild] mem0 unavailable; running in local-only (lexical) mode.")

    async def _diag_embed(self, text: str) -> Tuple[bool, Optional[Any]]:
        if self._mem_disabled or self._needs_async_rebuild:
            return False, None
        try:
            emb = await asyncio.to_thread(self._mem.embedding_model.embed, text, "add")
            logger.debug(f"[diag] embed result: {_describe(emb)}")
            ok = isinstance(emb, (list, tuple)) and (len(emb) > 0)
            return ok, emb
        except Exception as e:
            logger.error(f"[diag] embed failed: {e}")
            return False, None

    async def _diag_vector_search(self, query: str, emb: Any, user_id: str) -> None:
        if self._mem_disabled or self._needs_async_rebuild:
            return
        try:
            vs = self._mem.vector_store
            res = await asyncio.to_thread(
                vs.search, query=query, vectors=emb, limit=3, filters={"user_id": user_id}
            )
            logger.debug(f"[diag] vector_store.search -> {_describe(res)}")
            if isinstance(res, list) and res:
                first = res[0]
                try:
                    logger.debug(
                        f"[diag] first item type={type(first).__name__}, "
                        f"id={getattr(first, 'id', None)}, "
                        f"score={getattr(first, 'score', None)}, "
                        f"payload_type={type(getattr(first, 'payload', None)).__name__}"
                    )
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"[diag] vector_store.search failed: {e}")

    def _dump_cfg(self) -> None:
        if self._mem_disabled:
            logger.debug("[diag] mem0 disabled (local-only mode).")
            return
        try:
            cfg = getattr(self._mem, "config", None)
            em = getattr(cfg, "embedder", None)
            vs = getattr(cfg, "vector_store", None)
            llm = getattr(cfg, "llm", None)
            logger.debug(f"[diag] embedder: provider={getattr(em,'provider',None)} cfg={getattr(em,'config',None)}")
            logger.debug(f"[diag] vector_store: provider={getattr(vs,'provider',None)} cfg={getattr(vs,'config',None)}")
            logger.debug(f"[diag] llm: provider={getattr(llm,'provider',None)} cfg={getattr(llm,'config',None)}")
        except Exception as e:
            logger.debug(f"[diag] dump cfg failed: {e}")

    async def _switch_provider(self, provider: str) -> None:
        """
        Switch providers explicitly (only if you enable *_FALLBACK flags).
        """
        try:
            os.environ["MEM0_EMBEDDER"] = provider
            cfg, prov, dims, col, llm_prov, qpath, embed_model = _build_cfg(provider)
            self._cfg = cfg
            self._embedder_provider = prov
            self._embed_dims = dims
            self._collection = col
            self._llm_provider = llm_prov
            self._qdrant_path = qpath
            self._embed_model = embed_model
            self._mem = await _from_config_async(cfg)
            self._needs_async_rebuild = False
            self._mem_disabled = False
            logger.warning(f"[rebuild] switched embedder -> {provider} (dims={dims}, collection={col})")
        except Exception as e:
            logger.exception(f"[rebuild] failed switching to provider={provider}: {e}")
            self._mem_disabled = True
            logger.warning("[rebuild] disabling mem0 (unreachable). Using local-only mode.")

    # ---------- writes ----------

    async def add_user_text(self, user_id: str, text: str, **meta: Any) -> None:
        await self._ensure_async_rebuild()
        meta = {"role": "user", **(meta or {})}
        trace_id = f"add_user_text:{int(time.time()*1e6)}"
        logger.debug(f"[{trace_id}] add_user_text start user_id={user_id!r} len={len(text)} infer=False meta={list(meta.keys())}")
        if self._mem_disabled:
            _local_append(user_id, "user", text)
            logger.debug(f"[{trace_id}] mem0 disabled; wrote to local fallback.")
            return
        tried_fallback = False
        try:
            await self._mem.add(text, user_id=user_id, metadata=meta, infer=False)
            logger.debug(f"[{trace_id}] add_user_text mem0 OK")
            return
        except Exception as e:
            if (
                _is_auth_error(e)
                and self._ollama_fallback_enabled
                and self._embedder_provider == "openai"
            ):
                await self._switch_provider("ollama")
                tried_fallback = True
                try:
                    await self._mem.add(text, user_id=user_id, metadata=meta, infer=False)
                    logger.debug(f"[{trace_id}] add_user_text mem0 OK after ollama fallback")
                    return
                except Exception as e2:
                    logger.exception(f"[{trace_id}] mem0 add after fallback failed: {e2}")

            logger.error(f"[{trace_id}] mem0 add_user_text failed: {e}")
            self._dump_cfg()
            ok, emb = await self._diag_embed(text)
            if ok and emb is not None:
                await self._diag_vector_search(text, emb, user_id)
            _local_append(user_id, "user", text)
            logger.debug(f"[{trace_id}] add_user_text fell back to local store (tried_fallback={tried_fallback})")

    async def add_agent_text(self, user_id: str, text: str, **meta: Any) -> None:
        await self._ensure_async_rebuild()
        meta = {"role": "assistant", **(meta or {})}
        trace_id = f"add_agent_text:{int(time.time()*1e6)}"
        logger.debug(f"[{trace_id}] add_agent_text start user_id={user_id!r} len={len(text)} infer=False meta={list(meta.keys())}")
        if self._mem_disabled:
            _local_append(user_id, "assistant", text)
            logger.debug(f"[{trace_id}] mem0 disabled; wrote to local fallback.")
            return
        tried_fallback = False
        try:
            await self._mem.add(f"Agent: {text}", user_id=user_id, metadata=meta, infer=False)
            logger.debug(f"[{trace_id}] add_agent_text mem0 OK")
            return
        except Exception as e:
            if (
                _is_auth_error(e)
                and self._ollama_fallback_enabled
                and self._embedder_provider == "openai"
            ):
                await self._switch_provider("ollama")
                tried_fallback = True
                try:
                    await self._mem.add(f"Agent: {text}", user_id=user_id, metadata=meta, infer=False)
                    logger.debug(f"[{trace_id}] add_agent_text mem0 OK after ollama fallback")
                    return
                except Exception as e2:
                    logger.exception(f"[{trace_id}] mem0 add after fallback failed: {e2}")

            logger.error(f"[{trace_id}] mem0 add_agent_text failed: {e}")
            self._dump_cfg()
            ok, emb = await self._diag_embed(text)
            if ok and emb is not None:
                await self._diag_vector_search(text, emb, user_id)
            _local_append(user_id, "assistant", text)
            logger.debug(f"[{trace_id}] add_agent_text fell back to local store (tried_fallback={tried_fallback})")

    # ---------- search ----------

    async def _fallback_search(self, user_id: str, query: str, limit: int) -> List[str]:
        texts = _local_load_texts(user_id)
        if not texts:
            return []

        def score(a: str, b: str) -> float:
            try:
                return SequenceMatcher(None, a.lower(), b.lower()).ratio()
            except Exception:
                return 0.0

        ranked = sorted(((score(query, t), t) for t in texts), key=lambda x: x[0], reverse=True)
        return [t for _, t in ranked[:max(1, limit)]]

    async def search(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
        *,
        threshold: Optional[float] = None,
    ) -> List[str]:
        await self._ensure_async_rebuild()
        trace_id = f"search:{int(time.time()*1e6)}"
        logger.debug(f"[{trace_id}] mem0.search start user_id={user_id!r} limit={limit} threshold={threshold} q={query!r}")

        if self._mem_disabled:
            fb = await self._fallback_search(user_id, query, limit=limit)
            logger.debug(f"[{trace_id}] mem0 disabled; lexical results={len(fb)}")
            return fb

        tried_fallback = False
        try:
            res = await self._mem.search(
                query=query,
                user_id=user_id,
                limit=limit,
                threshold=threshold,
            )
            logger.debug(f"[{trace_id}] mem0.search OK payload={_describe(res)}")
            out = _normalize_results(res)
            logger.debug(f"[{trace_id}] normalized={len(out)} items")
            return out
        except Exception as e:
            # Optional provider switch if you started on OpenAI and want to auto-switch to Ollama on auth errors
            if (
                _is_auth_error(e)
                and self._ollama_fallback_enabled
                and self._embedder_provider == "openai"
            ):
                await self._switch_provider("ollama")
                tried_fallback = True
                try:
                    res = await self._mem.search(
                        query=query,
                        user_id=user_id,
                        limit=limit,
                        threshold=threshold,
                    )
                    logger.debug(f"[{trace_id}] mem0.search OK after ollama fallback payload={_describe(res)}")
                    out = _normalize_results(res)
                    logger.debug(f"[{trace_id}] normalized={len(out)} items (after fallback)")
                    return out
                except Exception as e2:
                    logger.exception(f"[{trace_id}] mem0 search after fallback failed: {e2}")

            logger.error(f"[{trace_id}] mem0 search failed: {e}")
            self._dump_cfg()

            ok, emb = await self._diag_embed(query)
            if ok and emb is not None:
                await self._diag_vector_search(query, emb, user_id)

            fb = await self._fallback_search(user_id, query, limit=limit)
            logger.debug(f"[{trace_id}] fallback lexical results={len(fb)} tried_fallback={tried_fallback}")
            return fb
