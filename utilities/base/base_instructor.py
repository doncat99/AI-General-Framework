# utilities/base/base_instructor.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

import instructor
from pydantic import BaseModel

from config import config
from .base_llm import BaseLLM
from .prompt_manager import PromptVars  # <- moved here (decoupled from base_llm)


class BaseInstructor(BaseLLM):
    """
    Thin Instructor wrapper over BaseLLM.

    - Reuses ALL prompt-management, param-precedence, and tracing logic from BaseLLM.
    - Only difference: we patch the underlying AsyncOpenAI client with `instructor.apatch`,
      so `chat.completions.create(...)` returns a **Pydantic model instance** when
      `response_model=SomeSchema` is provided.
    """

    def __init__(
        self,
        use_cache: bool = True,
        cache_dir: str = ".cache",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        instructor_mode: instructor.Mode = instructor.Mode.TOOLS,  # TOOLS or JSON
    ):
        # Initialize BaseLLM (client, tracing, etc.)
        super().__init__(
            use_cache=use_cache,
            cache_dir=cache_dir,
            api_key=api_key or config.OPENROUTER_API_KEY,
            base_url=base_url or config.OPENROUTER_BASE_URL,
        )
        # Swap in an Instructor-patched client. All upstream logic in BaseLLM remains intact.
        self.client = instructor.apatch(self.client, mode=instructor_mode)

    async def extract(
        self,
        response_model: Type[BaseModel],                     # REQUIRED by Instructor
        messages: Optional[List[Dict[str, Any]]] = None,     # optional; PromptVars may supply the content
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        app_name: str = "default_app",
        user_id: str = "default_user",
        session_id: str = "default_session",
        reasoning_effort: Optional[str] = "medium",          # minimal|low|medium|high if the model supports it
        prompt_vars: Optional[PromptVars] = None,            # single source of truth for prompt controls + vars
        **extra,
    ) -> Optional[BaseModel]:
        """
        Structured extraction via Instructor, fully reusing BaseLLM.extract:

        - BaseLLM handles: Langfuse Prompt fetch/compile, variable injection,
          param precedence (args > prompt.config > defaults), tracing, and linking
          the generation to the prompt version (if `link_prompt=True` in PromptVars).
        - Because `self.client` is instructor-patched, the underlying call returns
          a Pydantic `response_model` instance instead of a raw ChatCompletion.

        Returns:
            Pydantic model instance on success; None on failure.
        """
        # Just forward to BaseLLM.extract and add `response_model` to the OpenAI kwargs.
        # The instructor-patched client expects/consumes it.
        return await super().extract(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            reasoning_effort=reasoning_effort,
            prompt_vars=prompt_vars,
            response_model=response_model,  # <- consumed by instructor client
            **extra,
        )
