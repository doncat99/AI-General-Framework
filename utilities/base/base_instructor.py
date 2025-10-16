# utilities/base/base_instructor.py
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Type

from loguru import logger
import instructor
from pydantic import BaseModel

from config import config, langfuse
from .base_llm import BaseLLM, pickle_cache


class BaseInstructor(BaseLLM):
    """
    LLM client with Instructor support (structured outputs).
    - Inherits config, caching & tracing from BaseLLM.
    - Replaces `self.client` with an Instructor-patched Async client.
    - `extract` returns a Pydantic `response_model` instance (or None on failure).
    """

    def __init__(
        self,
        use_cache: bool = True,
        cache_dir: str = ".cache",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        instructor_mode: instructor.Mode = instructor.Mode.TOOLS,  # TOOLS or JSON
    ):
        super().__init__(
            use_cache=use_cache,
            cache_dir=cache_dir,
            api_key=api_key or config.OPEN_ROUTER_API_KEY,
            base_url=base_url or config.OPEN_ROUTER_API_BASE,
        )
        # Patch the ALREADY-initialized AsyncOpenAI client
        # apatch returns an Async client; no need to call patch() again
        self.client = instructor.apatch(self.client, mode=instructor_mode)

    def _from_dict(self, data: Dict[str, Any]) -> Any:
        """
        For cached results. Base impl returns dict. If you enable caching on
        `extract`, you should reconstruct a Pydantic model here OR in the caller.
        We leave it as dict to avoid coupling to a specific schema.
        """
        return data

    # Keep caching OFF by default unless you also persist the schema type.
    # @pickle_cache(cache_subdir="instructor_extract")
    async def extract(
        self,
        response_model: Type[BaseModel],         # required for Instructor
        messages: List[Dict[str, Any]],
        model: str,
        max_tokens: int,
        temperature: float = 0.0,
        app_name: str = "default_app",
        user_id: str = "default_user",
        session_id: str = "default_session",
        reasoning_effort: Optional[str] = "medium",  # minimal|low|medium|high if supported
        **extra,
    ) -> Optional[BaseModel]:
        """
        Structured extraction via Instructor.
        Returns a Pydantic model instance on success; None on failure.
        """
        result = None
        try:
            with langfuse.start_as_current_span(name=app_name) as span:
                kwargs = dict(
                    model=model,
                    response_model=response_model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                if reasoning_effort is not None:
                    kwargs["reasoning_effort"] = reasoning_effort
                if extra:
                    kwargs.update(extra)

                result = await self.client.chat.completions.create(**kwargs)

                # Trace (serialize conservatively)
                try:
                    span.update_trace(
                        input=messages,
                        output=_safe_dump_model(result),
                        user_id=user_id,
                        session_id=session_id,
                        tags=["agent", "instructor"],
                        metadata={"provider": "openai", "instructor_mode": "TOOLS"},
                        version="1.0.0",
                    )
                except Exception:
                    pass

            langfuse.flush()
        except Exception as e:
            logger.error(f"Error running extract: {e}")

        return result


def _safe_dump_model(m: Any) -> Any:
    try:
        if hasattr(m, "model_dump"):
            return m.model_dump()
        return json.loads(json.dumps(m, default=str))
    except Exception:
        try:
            return str(m)
        except Exception:
            return "<unserializable>"
