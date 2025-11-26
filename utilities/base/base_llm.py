# utilities/base/base_llm.py
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from functools import wraps
import asyncio
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic.version import VERSION as PYD_VER
from langfuse.openai import AsyncOpenAI  # OpenAI client wrapped with Langfuse

from config import config, langfuse  # your initialized Langfuse client + runtime config
from utilities.base.prompt_manager import PromptManager, PromptVars

# --- hard guard: pydantic v2 only ---
if not PYD_VER.startswith("2."):
    raise RuntimeError(f"Pydantic v2 is required; detected {PYD_VER}")


# -------------------------------
# Disk cache decorator (optional)
# -------------------------------

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
                    except Exception as e:  # pragma: no cover
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
                            payload = result
                        with open(cache_file, "w", encoding="utf-8") as f:
                            json.dump(payload, f, default=str, indent=2, ensure_ascii=False)
                    except Exception as e:  # pragma: no cover
                        logger.debug(f"[cache] skip write ({cache_file}): {e}")
                return result

        return wrapper

    return decorator


# --------------
# Base LLM class
# --------------

class BaseLLM:
    """
    Minimal async LLM base with Langfuse tracing + Prompt Management support.

    - Provides a Langfuse-instrumented OpenAI-compatible client (via langfuse.openai.AsyncOpenAI).
    - `extract` can fetch + compile a Langfuse-managed prompt using `PromptVars` (controls + variables).
    - The generation is automatically linked to the prompt version in Langfuse (if `link_prompt=True`).

    Parameter precedence for generation:
        explicit args (if not None) > prompt.config (from Langfuse) > internal defaults
    """

    def __init__(
        self,
        use_cache: bool = False,
        cache_dir: str = ".cache",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        # Raw client (OpenAI-compatible via OpenRouter settings)
        self.client = AsyncOpenAI(
            api_key=api_key or config.OPENROUTER_API_KEY,
            base_url=base_url or config.OPENROUTER_BASE_URL,
        )
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self._lock = asyncio.Lock()

        # Langfuse client (Prompt Management + manual spans)
        self.lf = langfuse
        self.pm = PromptManager(self.lf)

    # ---------- utils ----------
    def arg_hash(self, *args, **kwargs) -> str:
        try:
            key_bytes = json.dumps((args, kwargs), sort_keys=True, default=str).encode()
        except Exception:  # pragma: no cover
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
        messages: Optional[List[Dict[str, Any]]] = None,
        # Model params are Optional so prompt.config can fill them if caller leaves them as None
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        app_name: str = "default_app",
        user_id: str = "default_user",
        session_id: str = "default_session",
        reasoning_effort: Optional[str] = None,  # some models honor this
        # Unified PromptVars (controls + vars) from prompt_manager
        prompt_vars: Optional[PromptVars] = None,
        # passthrough extras to OpenAI chat.completions.create
        **extra,
    ) -> Any:
        """
        Raw Chat Completions call (async) with optional Langfuse Prompt Management.

        - If `prompt_vars.prompt_name` is set, we compile via PromptManager and *replace*
          `messages` with the resolved system+user pair from the managed prompt.
          (Templates typically use `{{instruction}}` and `{{contents}}`—we pass those via PromptVars.)
        - If the managed prompt is missing or PM is disabled, we fall back to provided `messages`.
        - Generation param precedence: explicit args (if not None) > prompt.config > defaults.
        """
        # Defaults
        DEFAULT_MODEL = "openai/gpt-4o-mini"
        DEFAULT_TEMPERATURE = 0.0
        DEFAULT_MAX_TOKENS = None  # let the client/model decide

        # 1) Managed prompt path (preferred if configured)
        lf_prompt_obj = None
        lf_cfg: Dict[str, Any] = {}

        if prompt_vars and prompt_vars.prompt_name:
            # We pass empty user_message_text because the template usually emits its own user turn
            resolved_instruction, resolved_user, lf_prompt_obj, lf_cfg = self.pm.compile_prompt(
                prompt_vars,
                fallback_instruction=None,
                user_message_text="",
            )
            managed_messages: List[Dict[str, str]] = []
            if resolved_instruction.strip():
                managed_messages.append({"role": "system", "content": resolved_instruction.strip()})
            if resolved_user.strip():
                managed_messages.append({"role": "user", "content": resolved_user.strip()})

            # If PM returned anything usable, we override messages with it
            if managed_messages:
                messages = managed_messages

        # 2) Ensure messages is a list (possibly empty)
        messages = messages or []

        # 3) Resolve effective generation params with precedence
        eff_model = model or lf_cfg.get("model") or DEFAULT_MODEL
        eff_temperature = temperature if temperature is not None else lf_cfg.get("temperature", DEFAULT_TEMPERATURE)
        eff_max_tokens = max_tokens or int(lf_cfg.get("max_tokens", DEFAULT_MAX_TOKENS))

        # 4) Build kwargs for the LLM call
        kwargs: Dict[str, Any] = dict(
            model=eff_model,
            messages=messages,
            max_tokens=eff_max_tokens,
            temperature=eff_temperature,
        )
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        if extra:
            kwargs.update(extra)

        # Link generation to the prompt version if requested
        if (prompt_vars is not None) and prompt_vars.link_prompt and lf_prompt_obj is not None and "langfuse_prompt" not in kwargs:
            kwargs["langfuse_prompt"] = lf_prompt_obj

        # 5) Call the model with a tracing span (SYNC context manager — fixes your _AgnosticContextManager error)
        response = None
        try:
            span_ctx = self.lf.start_as_current_span(name=app_name) if self.lf else None
            if span_ctx:
                with span_ctx as span:  # <-- sync context manager inside async fn is fine
                    response = await self.client.chat.completions.create(**kwargs)
                    try:  # best-effort structured trace
                        span.update_trace(
                            input=messages,
                            output=_safe_to_json(response),
                            user_id=user_id,
                            session_id=session_id,
                            tags=["agent", "llm"],
                            metadata={"provider": "openai", "model": eff_model},
                            version="1.0.0",
                        )
                    except Exception:
                        pass
            else:
                response = await self.client.chat.completions.create(**kwargs)

            try:
                if self.lf:
                    self.lf.flush()
            except Exception:
                pass
        except Exception as e:
            logger.error(f"[BaseLLM.extract] error: {e}")
        return response

    async def extract_text(self, *args, **kwargs) -> str:
        """Convenience: same args as `extract`, returns first choice text content (or "")."""
        resp = await self.extract(*args, **kwargs)

        if resp and getattr(resp, "choices", None):
            msg = resp.choices[0].message.content if resp.choices else ""
            return msg.strip() if isinstance(msg, str) else (msg or "")

        return ""


# -----------------
# JSON-safe helpers
# -----------------

def _safe_to_json(obj: Any) -> Any:
    try:
        return json.loads(json.dumps(obj, default=_repr_fallback))
    except Exception:  # pragma: no cover
        return _repr_fallback(obj)


def _repr_fallback(o: Any) -> str:  # pragma: no cover
    try:
        return repr(o)
    except Exception:
        return "<unserializable>"
