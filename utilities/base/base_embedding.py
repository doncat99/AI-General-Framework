# utilities/base/base_embedding.py
from __future__ import annotations

import asyncio
import math
import random
from typing import Any, Dict, List, Optional, Sequence

from loguru import logger
from langfuse.openai import AsyncOpenAI  # OpenAI-compatible, instrumented by Langfuse

from config import config, langfuse  # initialized Langfuse client + runtime config
from utilities.base.base_llm import pickle_cache  # reuse the JSON disk cache decorator


def _l2_normalize(vec: Sequence[float]) -> List[float]:
    """Return an L2-normalized copy of vec; zero vector returns zeros."""
    s = 0.0
    for x in vec:
        s += float(x) * float(x)
    if s <= 0.0:
        return [0.0 for _ in vec]
    inv = 1.0 / math.sqrt(s)
    return [float(x) * inv for x in vec]


def _fit_dim(vec: List[float], target: Optional[int]) -> List[float]:
    """Pad/truncate embedding to `target` dims if provided, else return as-is."""
    if target is None:
        return vec
    t = int(target)
    if t <= 0:
        return []
    if len(vec) == t:
        return vec
    if len(vec) > t:
        return vec[:t]
    return vec + [0.0] * (t - len(vec))


class BaseEmbedding:
    """
    Async embedding client with two backends:
      - backend="remote": OpenAI-compatible embeddings via AsyncOpenAI (Langfuse-wrapped)
      - backend="local": SentenceTransformer('all-MiniLM-L6-v2') (lazy-loaded)

    Features:
      - L2 normalization (optional)
      - Batching + retry with exponential backoff (remote)
      - Langfuse tracing per high-level call
      - JSON-on-disk caching via @pickle_cache
      - Rate limiting via asyncio.Semaphore
      - Async-callable instances (__call__)

    Usage:
        emb = BaseEmbedding(use_cache=True, cache_dir=".cache/emb", rate_limit=10)
        v = await emb("hello", backend="remote", normalize=True)
        v_local = await emb("hello", backend="local")
        vs = await emb(["a", "b"], backend="remote", dimensions=1536)
    """

    def __init__(
        self,
        *,
        use_cache: bool = False,
        cache_dir: str = ".cache",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
        default_dimensions: Optional[int] = None,
        sbert_model_name: Optional[str] = None,  # override local model name if needed
        max_batch_size: int = 512,
        max_retries: int = 3,
        initial_backoff_s: float = 0.5,
        backoff_jitter_s: float = 0.25,
        rate_limit: int = 20,
    ):
        # Remote (OpenAI-compatible)
        self.client = AsyncOpenAI(
            api_key=api_key or config.OPENAI_API_KEY,
            base_url=base_url or config.OPENAI_BASE_URL,
        )
        # Defaults (model/dimensions) can be wired from your config
        self.default_model = (
            default_model
            if (default_model is not None)
            else (getattr(config, "EMBEDDING_MODEL", None) or "openai/text-embedding-3-large")
        )
        self.default_dimensions = default_dimensions  # Applies to both backends when explicitly set

        # Local (SBERT) lazy-load setup
        self._sbert_model = None
        self.sbert_model_name = sbert_model_name or getattr(config, "SBERT_MODEL", "all-MiniLM-L6-v2")

        # Behavior knobs
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self._lock = asyncio.Lock()

        # Behavior knobs
        self.max_batch_size = max(1, int(max_batch_size))
        self.max_retries = max(0, int(max_retries))
        self.initial_backoff_s = float(initial_backoff_s)
        self.backoff_jitter_s = float(backoff_jitter_s)

        # Langfuse client for tracing
        self.lf = langfuse

        # NEW: simple concurrency cap — protects when multiple coroutines call this class
        self.rate_limit = max(1, int(rate_limit))
        self.semaphore = asyncio.Semaphore(self.rate_limit)

    # -------------
    # SBERT loader
    # -------------
    async def _ensure_sbert(self):
        """Lazy-load SentenceTransformer in a thread to avoid blocking the loop."""
        if self._sbert_model is not None:
            return
        try:
            # Import lazily to avoid hard dep when unused
            from sentence_transformers import SentenceTransformer  # type: ignore

            def _load():
                return SentenceTransformer(self.sbert_model_name)

            self._sbert_model = await asyncio.to_thread(_load)
            logger.info(f"Loaded SentenceTransformer model: {self.sbert_model_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to load SentenceTransformer '{self.sbert_model_name}': {e}")

    # -------------
    # Callable API
    # -------------
    async def __call__(
        self,
        input_texts: str | List[str],
        *,
        backend: str = "remote",
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
        normalize: bool = False,
        app_name: str = "embedding_app",
        user_id: str = "default_user",
        session_id: str = "default_session",
        **extra: Any,
    ) -> List[float] | List[List[float]]:
        if isinstance(input_texts, str):
            return await self.embed_query(
                input_texts,
                backend=backend,
                model=model,
                dimensions=dimensions,
                normalize=normalize,
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
                **extra,
            )
        elif isinstance(input_texts, list):
            return await self.embed_texts(
                input_texts,
                backend=backend,
                model=model,
                dimensions=dimensions,
                normalize=normalize,
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
                **extra,
            )
        else:
            raise TypeError("input_texts must be str or List[str]")

    async def embed(self, input_texts: str | List[str], **kwargs: Any):
        return await self.__call__(input_texts, **kwargs)

    # -------------
    # Public API
    # -------------
    @pickle_cache(cache_subdir="embeddings_query")
    async def embed_query(
        self,
        text: str,
        *,
        backend: str = "remote",
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
        normalize: bool = False,
        app_name: str = "embedding_app",
        user_id: str = "default_user",
        session_id: str = "default_session",
        **extra: Any,
    ) -> List[float]:
        """Embed a single query string. Returns a single vector (possibly normalized)."""
        vectors = await self.embed_texts(
            [text],
            backend=backend,
            model=model,
            dimensions=dimensions,
            normalize=normalize,
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            **extra,
        )
        return vectors[0] if vectors else []

    @pickle_cache(cache_subdir="embeddings_texts")
    async def embed_texts(
        self,
        texts: List[str],
        *,
        backend: str = "remote",
        model: Optional[str] = None,          # used for remote
        dimensions: Optional[int] = None,     # if set, pad/truncate for both backends
        normalize: bool = False,
        app_name: str = "embedding_app",
        user_id: str = "default_user",
        session_id: str = "default_session",
        **extra: Any,
    ) -> List[List[float]]:
        """
        Embed a list of strings, preserving order. Splits into batches and retries transient errors.
        """
        if not texts:
            return []

        eff_model = model or self.default_model
        eff_dims = dimensions if (dimensions is not None) else self.default_dimensions
        backend = (backend or "remote").lower()
        if backend not in ("remote", "local"):
            raise ValueError("backend must be 'remote' or 'local'")

        # Pre-allocate results in order
        results: List[Optional[List[float]]] = [None] * len(texts)

        # Trace entire call (not individual batches) to avoid noisy traces
        span_ctx = self.lf.start_as_current_span(name=app_name) if self.lf else None
        try:
            if span_ctx:
                with span_ctx as span:
                    await self._embed_batched(
                        texts=texts,
                        backend=backend,
                        eff_model=eff_model,
                        eff_dims=eff_dims,
                        normalize=normalize,
                        dest=results,
                        extra=extra,
                    )
                    try:
                        span.update_trace(
                            input={"count": len(texts), "backend": backend, "model": eff_model if backend == "remote" else self.sbert_model_name, "dimensions": eff_dims},
                            output={"ok": sum(1 for v in results if v is not None), "failed": sum(1 for v in results if v is None)},
                            user_id=user_id,
                            session_id=session_id,
                            tags=["embedding", backend],
                            metadata={"provider": "openai" if backend == "remote" else "local-sbert"},
                            version="1.1.0",
                        )
                    except Exception:
                        pass
            else:
                await self._embed_batched(
                    texts=texts,
                    backend=backend,
                    eff_model=eff_model,
                    eff_dims=eff_dims,
                    normalize=normalize,
                    dest=results,
                    extra=extra,
                )

            # Fill missing as empty-fitted vectors to keep shape stable
            filled: List[List[float]] = []
            for v in results:
                vec = v if v is not None else _fit_dim([], eff_dims)
                filled.append(vec)
            return filled
        finally:
            try:
                if self.lf:
                    self.lf.flush()
            except Exception:
                pass

    # -------------
    # Internals
    # -------------
    async def _embed_batched(
        self,
        *,
        texts: List[str],
        backend: str,
        eff_model: str,
        eff_dims: Optional[int],
        normalize: bool,
        dest: List[Optional[List[float]]],
        extra: Dict[str, Any],
    ) -> None:
        """
        Split into request batches, run sequentially to simplify rate-limit handling.
        Fills `dest` in-place at positions matching the original input order.
        """
        n = len(texts)
        idx = 0
        while idx < n:
            j = min(idx + self.max_batch_size, n)
            batch = texts[idx:j]
            try:
                if backend == "remote":
                    vectors = await self._embed_call_remote_with_retry(
                        batch,
                        model=eff_model,
                        dimensions=eff_dims,
                        **extra,
                    )
                else:
                    vectors = await self._embed_call_local(batch)

                if normalize:
                    vectors = [_l2_normalize(v) for v in vectors]
                if eff_dims is not None:
                    vectors = [_fit_dim(v, eff_dims) for v in vectors]

                for k, vec in enumerate(vectors):
                    dest[idx + k] = vec
            except Exception as e:
                logger.error(f"[embedding/{backend}] batch {idx}:{j} failed: {e}")
            idx = j

    async def _embed_call_remote_with_retry(
        self,
        batch: List[str],
        *,
        model: str,
        dimensions: Optional[int],
        **extra: Any,
    ) -> List[List[float]]:
        """
        Single API call with bounded retries. Returns embeddings for `batch` in order.
        Rate-limited via semaphore to cap concurrent in-flight requests.
        """
        attempt = 0
        last_err: Optional[Exception] = None

        while attempt <= self.max_retries:
            try:
                kwargs: Dict[str, Any] = {"model": model, "input": batch}
                if dimensions is not None:
                    kwargs["dimensions"] = int(dimensions)
                if extra:
                    kwargs.update(extra)

                # <= NEW: concurrency cap
                async with self.semaphore:
                    resp = await self.client.embeddings.create(**kwargs)

                return [list(item.embedding) for item in resp.data]
            except Exception as e:
                last_err = e
                # backoff
                if attempt >= self.max_retries:
                    break
                sleep_s = self.initial_backoff_s * (2 ** attempt) + random.uniform(0.0, self.backoff_jitter_s)
                await asyncio.sleep(sleep_s)
                attempt += 1

        raise RuntimeError(f"Remote embedding failed after {self.max_retries + 1} attempts: {last_err}")

    async def _embed_call_local(self, batch: List[str]) -> List[List[float]]:
        await self._ensure_sbert()

        # SBERT encode is sync; run in a thread
        def _encode():
            # convert_to_numpy=True ensures np.ndarray; normalize handled separately
            embs = self._sbert_model.encode(batch, convert_to_numpy=True, normalize_embeddings=False)
            # Handle single/2D shape variations across versions
            if embs is None:
                return []
            if len(getattr(embs, "shape", [])) == 1:
                return [embs.tolist()]
            return [row.tolist() for row in embs]

        async with self.semaphore:
            vectors: List[List[float]] = await asyncio.to_thread(_encode)
        return vectors
