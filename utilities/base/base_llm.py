# utilities/base/base_llm.py
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from functools import wraps
import asyncio
from typing import Any, Dict, List, Optional

from loguru import logger
from langfuse.openai import AsyncOpenAI  # OpenAI client wrapped with Langfuse

from config import config, langfuse


def pickle_cache(cache_subdir: str):
    """
    Decorator to cache async function results to JSON on disk.
    NOTE:
      - Uses self.use_cache / self.cache_dir / self._lock from the instance.
      - Stores the *returned value* via .model_dump() if it has it, else json.dump() of dict.
      - On load, it calls self._from_dict(data) to rehydrate (override in subclasses if needed).
      - Keep it OFF for now if your function returns non-serializable types (e.g., ChatCompletion).
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            if not getattr(self, "use_cache", False) or not getattr(self, "cache_dir", "") or not getattr(self, "_lock", None):
                return await func(self, *args, **kwargs)

            cache_dir = Path(self.cache_dir) / cache_subdir
            cache_dir.mkdir(parents=True, exist_ok=True)

            key = self.arg_hash(func.__name__, *args, **kwargs)
            cache_file = cache_dir / f"{key}.json"

            async with self._lock:
                if cache_file.exists():
                    try:
                        with open(cache_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        return self._from_dict(data)
                    except Exception as e:
                        logger.warning(f"[cache] Failed to read cache {cache_file}: {e}")

                result = await func(self, *args, **kwargs)
                if result is not None:
                    try:
                        # best-effort serialization
                        if hasattr(result, "model_dump"):
                            payload = result.model_dump()
                        elif isinstance(result, dict):
                            payload = result
                        else:
                            # fall back: try to json.dumps it
                            json.dumps(result, default=str)  # test
                            payload = result  # might be a dict-like
                        with open(cache_file, "w", encoding="utf-8") as f:
                            json.dump(payload, f, default=str, indent=2, ensure_ascii=False)
                    except Exception as e:
                        logger.debug(f"[cache] skip write ({cache_file}): {e}")
                return result
        return wrapper
    return decorator


class BaseLLM:
    """
    Minimal async LLM base — NO Instructor.
    - Provides a Langfuse-instrumented OpenAI-compatible client.
    - Implements a generic `extract` that returns the raw ChatCompletion.
    - Add helper `extract_text` to get first message content.
    """

    def __init__(
        self,
        use_cache: bool = False,
        cache_dir: str = ".cache",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        # Raw client (no Instructor patching here)
        self.client = AsyncOpenAI(
            api_key=api_key or config.OPEN_ROUTER_API_KEY,
            base_url=base_url or config.OPEN_ROUTER_BASE_URL,
        )
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self._lock = asyncio.Lock()

    # ---------- utils ----------
    def arg_hash(self, *args, **kwargs) -> str:
        try:
            key_bytes = json.dumps((args, kwargs), sort_keys=True, default=str).encode()
        except Exception:
            key_bytes = str((args, kwargs)).encode()
        return hashlib.sha256(key_bytes).hexdigest()

    def _from_dict(self, data: Dict[str, Any]) -> Any:
        """
        Rehydrate cached objects. Base impl returns dict as-is.
        Subclasses that return Pydantic models can override to reconstruct.
        """
        return data

    # ---------- main calls ----------
    # @pickle_cache(cache_subdir="llm_extract")  # keep off unless you rehydrate reliably
    async def extract(
        self,
        response_model: Optional[type] = None,     # ignored in base (no Instructor)
        messages: List[Dict[str, Any]] = None,
        model: str = "gpt-4o-mini",
        max_tokens: int = 4096,
        temperature: float = 0.0,
        app_name: str = "default_app",
        user_id: str = "default_user",
        session_id: str = "default_session",
        reasoning_effort: Optional[str] = None,   # "minimal" | "low" | "medium" | "high" (if supported by model)
        **extra,
    ) -> Any:
        """
        Raw OpenAI Chat Completions call (async). Returns ChatCompletion.
        """
        messages = messages or []
        response = None
        try:
            with langfuse.start_as_current_span(name=app_name) as span:
                kwargs = dict(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                if reasoning_effort is not None:
                    kwargs["reasoning_effort"] = reasoning_effort  # only some models honor this
                if extra:
                    kwargs.update(extra)

                response = await self.client.chat.completions.create(**kwargs)

                # Trace
                try:
                    span.update_trace(
                        input=messages,
                        output=_safe_to_json(response),
                        user_id=user_id,
                        session_id=session_id,
                        tags=["agent", "llm"],
                        metadata={"provider": "openai"},
                        version="1.0.0",
                    )
                except Exception:
                    pass
            langfuse.flush()
        except Exception as e:
            logger.error(f"[BaseLLM.extract] error: {e}")
        return response

    async def extract_text(
        self,
        *args,
        **kwargs,
    ) -> str:
        """
        Convenience: same args as `extract`, returns first choice text content (or "").
        """
        resp = await self.extract(*args, **kwargs)
        try:
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            return ""


def _safe_to_json(obj: Any) -> Any:
    try:
        return json.loads(json.dumps(obj, default=_repr_fallback))
    except Exception:
        return _repr_fallback(obj)


def _repr_fallback(o: Any) -> str:
    try:
        return repr(o)
    except Exception:
        return "<unserializable>"
