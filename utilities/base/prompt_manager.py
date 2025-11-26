# utilities/base/prompt_manager.py
from __future__ import annotations

"""
Prompt Manager
--------------
- Fetch & compile managed prompts (Langfuse).
- If missing, attempt to CREATE (SDK-shape aware), then re-fetch.
- Provide safe fallbacks when PM is disabled/unavailable.
- NEW: per-app default config (model/temperature/max_tokens) with env overrides.

Flow: get → ok → compile
      └─ not found → create → re-get → ok → compile
                       └─ fail → fallback

Kill-switch:
  export DISABLE_LANGFUSE_PM=1

Env override (JSON):
  export PM_APP_DEFAULTS_JSON='{"ACCA_SUSTAINABILITY_REPORTING":{"model":"anthropic/claude-sonnet-4","max_tokens":120000}}'
"""

import os
import json
import inspect
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, List, cast

from loguru import logger
from pydantic import BaseModel, Field, AliasChoices, ConfigDict, field_validator

try:
    from config import langfuse  # project-level Langfuse client or None
except Exception:
    langfuse = None  # type: ignore


# -------------------- Prompt Vars --------------------

class PromptVars(BaseModel):
    """
    Unified prompt variables passed to the prompt manager.

    Keys:
      - fetch_type: how to fetch from PM backend ('chat' or 'text').
      - apply: how to apply compiled pieces to the turn ('system'|'user'|'both').

    Back-compat aliasing:
      - 'type' or 'fetch_type' -> fetch_type
      - 'prompt_type'          -> apply
    """
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    # Lookup keys in PM
    prompt_name: Optional[str] = None
    prompt_label: Optional[str] = None
    prompt_version: Optional[str] = None

    # How to fetch from PM ('chat' or 'text')
    fetch_type: str = Field(
        default="chat",
        validation_alias=AliasChoices("fetch_type", "type"),
    )

    # How to apply compiled pieces ('system'|'user'|'both')
    apply: str = Field(
        default="both",
        validation_alias=AliasChoices("apply", "prompt_type"),
    )

    # Template variables exposed to compilation
    variables: Dict[str, Any] = Field(default_factory=dict)

    # Common convenience vars many templates use
    instruction: Optional[str] = None  # {{instruction}}
    contents: Optional[str] = None     # {{contents}}

    # Best-effort prompt linking in traces
    link_prompt: bool = True

    @field_validator("fetch_type")
    @classmethod
    def _norm_fetch_type(cls, v: str) -> str:
        v2 = (v or "chat").strip().lower()
        return "text" if v2 == "text" else "chat"

    @field_validator("apply")
    @classmethod
    def _norm_apply(cls, v: str) -> str:
        v2 = (v or "both").strip().lower()
        return v2 if v2 in {"system", "user", "both"} else "both"

    def model_dump_for_compile(self) -> Dict[str, Any]:
        """What we pass into Langfuse's compile(**kwargs)."""
        return {
            "variables": self.variables or {},
            "instruction": self.instruction,
            "contents": self.contents,
        }


# -------------------- Manager --------------------

DISABLE_PM: bool = (os.getenv("DISABLE_LANGFUSE_PM", "").strip().lower() in ("1", "true", "yes", "on"))


class PromptManager:
    """
    Facade that hides Langfuse specifics and provides:
      - existence probe
      - compile with fallbacks
      - on-miss auto-creation (best effort across SDK shapes)
    """

    def __init__(self, lf_client: Any = None) -> None:
        # Allow injection; default to project-wide config.langfuse
        self._lf = lf_client if lf_client is not None else langfuse

    # -------- env / status --------

    @property
    def enabled(self) -> bool:
        """True only if PM is not kill-switched and a client is available."""
        return (not DISABLE_PM) and (self._lf is not None)

    # -------- helpers --------

    @staticmethod
    def build_prompt_vars(
        *,
        prompt_name: Optional[str],
        prompt_label: Optional[str] = None,
        prompt_version: Optional[str] = None,
        fetch_type: str = "chat",         # 'chat'|'text'
        apply: str = "both",              # 'system'|'user'|'both'
        variables: Optional[Dict[str, Any]] = None,
        instruction: Optional[str] = None,
        contents: Optional[str] = None,
        link_prompt: bool = True,
    ) -> PromptVars:
        """Factory so callers don't depend on constructor details."""
        return PromptVars(  # type: ignore[call-arg]
            prompt_name=prompt_name,
            prompt_label=prompt_label,
            prompt_version=prompt_version,
            fetch_type=fetch_type,
            apply=apply,
            variables=variables or {},
            instruction=instruction,
            contents=contents,
            link_prompt=link_prompt,
        )

    @staticmethod
    def merge_vars(*dicts: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Left-to-right merge (later overwrites earlier)."""
        out: Dict[str, Any] = {}
        for d in dicts:
            if d:
                out.update(d)
        return out

    # -------- discovery --------

    def prompt_exists(
        self,
        prompt_name: str,
        *,
        label: Optional[str] = None,
        version: Optional[str] = None,
        prompt_type: str = "chat",
    ) -> bool:
        """
        Cheap "does it exist?" probe. Returns False on any failure.
        """
        if not self.enabled:
            return False
        try:
            kwargs: Dict[str, Any] = {"type": prompt_type}
            if version is not None:
                kwargs["version"] = version
            elif label:
                kwargs["label"] = label
            _ = self._lf.get_prompt(prompt_name, **kwargs)
            return True
        except Exception as e:
            logger.debug(f"[prompt_manager] '{prompt_name}' not found/usable: {e}")
            return False

    # -------- compile (get → maybe-create → get → fallback) --------

    def compile_prompt(
        self,
        pv: Optional[PromptVars],
        *,
        fallback_instruction: Optional[str],
        user_message_text: str,
    ) -> Tuple[str, str, Optional[Any], Dict[str, Any]]:
        """
        Returns:
          - resolved_instruction (system)
          - resolved_user_text  (user)
          - lf_prompt_obj       (opaque handle; may be None)
          - lf_config           (dict; may be empty)
        """
        # fallbacks
        resolved_instruction = (fallback_instruction or (pv.instruction if pv else "") or "").strip()
        resolved_user_text = user_message_text
        lf_prompt_obj: Optional[Any] = None
        lf_cfg: Dict[str, Any] = {}

        if not (pv and pv.prompt_name):
            return resolved_instruction, resolved_user_text, None, {}

        if not self.enabled:
            # PM kill-switched or client missing; no-op
            return resolved_instruction, resolved_user_text, None, {}

        # 1) Try fetch+compile
        ok, lf_prompt_obj, resolved_instruction, resolved_user_text, lf_cfg = \
            self._try_fetch_and_compile(pv, fallback_instruction, user_message_text)
        if ok:
            return resolved_instruction, resolved_user_text, lf_prompt_obj, lf_cfg

        # 2) Create (runtime-shape aware), then re-try
        if self.create_prompt(pv):
            ok, lf_prompt_obj, resolved_instruction, resolved_user_text, lf_cfg = \
                self._try_fetch_and_compile(pv, fallback_instruction, user_message_text)
            if ok:
                return resolved_instruction, resolved_user_text, lf_prompt_obj, lf_cfg

        # 3) Fallback
        return resolved_instruction, resolved_user_text, None, {}

    # -------- fetch + compile --------

    def _try_fetch_and_compile(
        self,
        pv: PromptVars,
        fallback_instruction: Optional[str],
        user_message_text: str,
    ) -> Tuple[bool, Optional[Any], str, str, Dict[str, Any]]:
        """
        Attempt a single fetch+compile; return (ok, prompt_obj, instr, user, cfg)
        """
        try:
            kwargs: Dict[str, Any] = {"type": pv.fetch_type}
            if pv.prompt_version is not None:
                kwargs["version"] = pv.prompt_version
            elif pv.prompt_label:
                kwargs["label"] = pv.prompt_label

            p = self._lf.get_prompt(pv.prompt_name, **kwargs)
            compiled = p.compile(**pv.model_dump_for_compile())

            # best-effort config exposure for tracing/audits
            try:
                lf_cfg = cast(Dict[str, Any], getattr(p, "config", {}) or {})
            except Exception:
                lf_cfg = {}

            instr, user = self._apply_compiled(pv, compiled, fallback_instruction, user_message_text)
            return True, p, instr, user, lf_cfg
        except Exception as e:
            logger.debug(f"[prompt_manager] fetch/compile failed for '{pv.prompt_name}': {e}")
            return False, None, (fallback_instruction or pv.instruction or "").strip(), user_message_text, {}

    # -------- creation (SDK-shape aware) --------

    def create_prompt(self, pv: PromptVars) -> bool:
        """
        Create/upsert a minimal prompt. Handles different Langfuse client shapes:
          - create_prompt(name=..., type='chat', prompt=[...], labels=[...], config={...})
          - create_prompt(name=..., type='chat', messages=[...], labels=[...], config={...})
          - create_prompt(name=..., type='text',  text='...', labels=[...],  config={...})
          - upsert_prompt(... same variants ...)
          - prompts.create(dict_payload)  or  prompts.create(**kwargs)
        """
        if not (self.enabled and pv and pv.prompt_name):
            return False

        label = pv.prompt_label or "production"
        # Minimal content scaffolds
        chat_messages = [
            {"role": "system", "content": "{{ instruction }}"},
            {"role": "user", "content": "{{ contents }}"},
        ]
        text_blob = "{{ instruction }}\n\n{{ contents }}"

        # Suggest some sane defaults; method may ignore if unsupported.
        common_cfg = {"model": "openai/gpt-4o", "temperature": 0, "max_tokens": 65536}

        # Helper to try a callable with kwargs discovered via signature
        def _try_callable(fn, kind: str) -> bool:
            try:
                sig = inspect.signature(fn)
                params = set(sig.parameters.keys())

                kwargs: Dict[str, Any] = {"name": pv.prompt_name, "type": pv.fetch_type, "labels": [label]}
                if "config" in params:
                    kwargs["config"] = common_cfg

                if pv.fetch_type == "chat":
                    if "prompt" in params:
                        kwargs["prompt"] = chat_messages
                    elif "messages" in params:
                        kwargs["messages"] = chat_messages
                    else:
                        # last resort: try single dict payload signature
                        pass
                else:  # text
                    if "text" in params:
                        kwargs["text"] = text_blob
                    elif "prompt" in params:
                        kwargs["prompt"] = text_blob
                    else:
                        pass

                # If fn seems to accept a single dict payload (e.g., prompts.create(payload))
                if len(params) == 1 and next(iter(params)).lower() in {"data", "payload", "body", "prompt"}:
                    payload = {
                        "name": pv.prompt_name,
                        "type": pv.fetch_type,
                        "labels": [label],
                        "config": common_cfg,
                    }
                    if pv.fetch_type == "chat":
                        payload["prompt"] = chat_messages
                    else:
                        payload["text"] = text_blob
                    fn(payload)  # type: ignore[misc]
                    return True

                fn(**kwargs)
                return True
            except Exception as e:
                logger.debug(f"[prompt_manager] {kind} failed: {e}")
                return False

        # 1) langfuse.create_prompt(...)
        if hasattr(self._lf, "create_prompt") and _try_callable(self._lf.create_prompt, "create_prompt"):
            logger.info(f"[prompt_manager] Created prompt via create_prompt: {pv.prompt_name}")
            return True

        # 2) langfuse.upsert_prompt(...)
        if hasattr(self._lf, "upsert_prompt") and _try_callable(self._lf.upsert_prompt, "upsert_prompt"):
            logger.info(f"[prompt_manager] Upserted prompt via upsert_prompt: {pv.prompt_name}")
            return True

        # 3) langfuse.prompts.create(...)
        prompts = getattr(self._lf, "prompts", None)
        if prompts is not None and hasattr(prompts, "create"):
            # Try dict-payload first (most SDKs use this), then kwargs fallback.
            payload = {
                "name": pv.prompt_name,
                "type": pv.fetch_type,
                "labels": [label],
                "config": common_cfg,
            }
            if pv.fetch_type == "chat":
                payload["prompt"] = chat_messages
            else:
                payload["text"] = text_blob

            # 3a) single-arg dict
            try:
                prompts.create(payload)  # type: ignore[misc]
                logger.info(f"[prompt_manager] Created prompt via prompts.create(data): {pv.prompt_name}")
                return True
            except Exception as e:
                logger.debug(f"[prompt_manager] prompts.create(data) failed: {e}")

            # 3b) kwargs
            try:
                prompts.create(**payload)  # type: ignore[misc]
                logger.info(f"[prompt_manager] Created prompt via prompts.create(**kwargs): {pv.prompt_name}")
                return True
            except Exception as e:
                logger.debug(f"[prompt_manager] prompts.create(**kwargs) failed: {e}")

        logger.warning(f"[prompt_manager] Could not create prompt '{pv.prompt_name}' with available client methods.")
        return False

    # -------- apply compiled content --------

    def _apply_compiled(
        self,
        pv: PromptVars,
        compiled: Any,
        fallback_instruction: Optional[str],
        user_message_text: str,
    ) -> Tuple[str, str]:
        """
        Respect pv.fetch_type ('chat'|'text') and pv.apply ('system'|'user'|'both').
        """
        resolved_instruction = (fallback_instruction or pv.instruction or "").strip()
        resolved_user_text = user_message_text

        if pv.fetch_type == "chat":
            sys_texts, user_texts = self._normalize_compiled_chat(compiled)

            if pv.apply in {"system", "both"}:
                sys_joined = "\n\n".join([t for t in sys_texts if t.strip()]).strip()
                if sys_joined:
                    resolved_instruction = sys_joined
                # explicit caller override still wins
                if fallback_instruction and fallback_instruction.strip():
                    resolved_instruction = fallback_instruction.strip()
                if not resolved_instruction:
                    resolved_instruction = (pv.instruction or "").strip()

            if pv.apply in {"user", "both"}:
                prefix_user = "\n\n".join([t for t in user_texts if t.strip()]).strip()
                if prefix_user:
                    resolved_user_text = f"{prefix_user}\n\n{user_message_text}".strip()

            return resolved_instruction, resolved_user_text

        # fetch_type == "text"
        compiled_text = self._as_text(compiled).strip()

        if pv.apply in {"system", "both"}:
            if compiled_text:
                resolved_instruction = compiled_text
            if fallback_instruction and fallback_instruction.strip():
                resolved_instruction = fallback_instruction.strip()
            if not resolved_instruction:
                resolved_instruction = (pv.instruction or "").strip()

        if pv.apply in {"user", "both"} and compiled_text:
            resolved_user_text = f"{compiled_text}\n\n{user_message_text}".strip()

        return resolved_instruction, resolved_user_text

    # -------- normalization helpers --------

    @staticmethod
    def _as_text(compiled: Any) -> str:
        """
        Normalize compiled prompt into a text blob (for 'text' prompts).
        Accepts str directly; if dict/list given, serialize minimalistically.
        """
        if isinstance(compiled, str):
            return compiled
        try:
            # Some Langfuse versions return dict with 'text'
            if isinstance(compiled, dict) and "text" in compiled:
                return str(compiled["text"])
        except Exception:
            pass
        # Last resort: JSON dump
        try:
            return json.dumps(compiled, ensure_ascii=False)
        except Exception:
            return str(compiled)

    @staticmethod
    def _normalize_compiled_chat(compiled: Any) -> Tuple[List[str], List[str]]:
        """
        Accept a variety of compiled chat shapes and normalize to:
            (system_texts: List[str], user_texts: List[str])
        Supported shapes:
            - List[{"role": "...", "content": "..."}]
            - {"messages": [...]}  (same inner shape)
            - {"system": "...", "user": "..."}   (legacy/simple)
            - str  (treated as single user message)
        """
        sys_texts: List[str] = []
        user_texts: List[str] = []

        try:
            # Common case: {'messages': [...]}
            if isinstance(compiled, dict) and "messages" in compiled and isinstance(compiled["messages"], list):
                msgs = compiled["messages"]
            elif isinstance(compiled, list):
                msgs = compiled
            elif isinstance(compiled, dict) and ("system" in compiled or "user" in compiled):
                # very simple dict form
                if "system" in compiled and isinstance(compiled["system"], str):
                    sys_texts.append(compiled["system"])
                if "user" in compiled and isinstance(compiled["user"], str):
                    user_texts.append(compiled["user"])
                return sys_texts, user_texts
            elif isinstance(compiled, str):
                # treat as a user preface
                user_texts.append(compiled)
                return sys_texts, user_texts
            else:
                # unknown shape → stringify once as a user preface
                user_texts.append(json.dumps(compiled, ensure_ascii=False))
                return sys_texts, user_texts

            # Parse list of messages
            for m in msgs:
                try:
                    role = (m.get("role") or "").strip().lower()
                    content = m.get("content")
                    if isinstance(content, list):
                        # e.g., OpenAI multipart; concatenate textual parts
                        parts = []
                        for part in content:
                            if isinstance(part, dict) and "text" in part:
                                parts.append(str(part["text"]))
                            elif isinstance(part, str):
                                parts.append(part)
                        content = "\n".join(parts)
                    text = str(content) if content is not None else ""
                    if role == "system":
                        sys_texts.append(text)
                    elif role == "user":
                        user_texts.append(text)
                except Exception:
                    continue
            return sys_texts, user_texts

        except Exception:
            # ultra-conservative fallback
            try:
                user_texts.append(json.dumps(compiled, ensure_ascii=False))
            except Exception:
                user_texts.append(str(compiled))
            return sys_texts, user_texts
