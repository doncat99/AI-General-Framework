# agents/assistant/agent.py
from __future__ import annotations
from typing import List, Optional, Any, Dict
from loguru import logger
import asyncio
import os
import time
import re
import inspect

from .memory import Memory
from .adk_client import ADKClient
from .prompts.prompt import get_system_prompt, build_user_with_memories

# New unified tool registry
from .tools.agent_tools import get_registry as get_tool_registry, register_with_agent

# Optional: external toolpacks (if your loader returns a compatible registry, we'll merge it)
try:
    from .tools.external_loader import load_external_tools  # may return a ToolRegistry or a plain dict/iterable
except Exception:
    load_external_tools = None  # type: ignore[assignment]

# RAG is optional; guard imports so the agent still works without it
try:
    # Keep .setting if that is your current project layout. Switch to .config if needed.
    from .setting import ENABLE_RAG as DEFAULT_ENABLE_RAG, CHROMA_DIR
    from .retrievers.chroma_retriever import ChromaRetriever
    from .rag import RAG
except Exception:
    DEFAULT_ENABLE_RAG = False  # type: ignore[assignment]
    CHROMA_DIR = ""             # type: ignore[assignment]
    RAG = None                  # type: ignore[assignment]
    ChromaRetriever = None      # type: ignore[assignment]


_SANITIZE_RE = re.compile(r"[^0-9A-Za-z_]+")


def _norm(n: str) -> str:
    return _SANITIZE_RE.sub("_", (n or "").strip())


# Known ADK internal/foundation tool aliases — skip duplicates from plugins
_ADK_INTERNAL_ALIASES = {
    "memory_upsert", "memory_search", "memory_delete",
    "rag_query",
    "web_get",
    "text_summarize",
    "math_eval",
    "files_list",
    "time_now",
}


def _register_tool_with_client(client: Any, name: str, handler: Any, description: str, parameters: Dict[str, Any]) -> None:
    """
    Register a function tool with ADK in a schema-first way (prevents fallback logs).
    Tries multiple client APIs for compatibility across ADK versions.
    """
    # Prefer modern API
    if hasattr(client, "add_function_tool"):
        client.add_function_tool(name, description, parameters, handler)
        return

    # Older or alternate API
    if hasattr(client, "register_tool"):
        try:
            client.register_tool(name, description, parameters, handler)
        except TypeError:
            client.register_tool(name=name, description=description, parameters=parameters, handler=handler)
        return

    # Absolute fallback
    add_tool = getattr(client, "add_tool", None)
    if callable(add_tool):
        add_tool(handler)
        return

    logger.warning(f"Could not register tool '{name}': no compatible ADK registration method.")


def _register_external_anyshape(client: Any, ext_obj: Any, used_norm_names: set[str]) -> List[str]:
    """
    Register external tools to ADK no matter how the loader formats them.
    Supports:
      - ToolRegistry-like (has .tools iterable of objs with name/handler/description/parameters)
      - Dict[str, callable]
      - Dict[str, {handler, description?, parameters?}]
      - Iterable[callable] (name via __name__)
    Skips names that collide with 'used_norm_names' (prevents duplicate alias warnings).
    Returns list of registered tool names.
    """
    registered: List[str] = []

    def _maybe_register(tool_name: str, handler: Any, description: str = "", parameters: Dict[str, Any] | None = None):
        if not callable(handler):
            return
        nm = _norm(tool_name)
        if nm in used_norm_names or nm in _ADK_INTERNAL_ALIASES:
            logger.debug(f"Skipping external tool '{tool_name}' (duplicate/internal).")
            return
        _register_tool_with_client(
            client,
            tool_name,
            handler,
            description or getattr(handler, "__doc__", "") or "",
            parameters or {"type": "object", "properties": {}, "required": []},
        )
        used_norm_names.add(nm)
        registered.append(str(tool_name))

    # 1) Registry-like shape
    if hasattr(ext_obj, "tools"):
        for t in getattr(ext_obj, "tools"):
            name = getattr(t, "name", None) or getattr(t, "__name__", None) or "tool"
            handler = getattr(t, "handler", None) or t
            description = getattr(t, "description", None) or getattr(handler, "__doc__", "") or ""
            parameters = getattr(t, "parameters", None) or {"type": "object", "properties": {}, "required": []}
            _maybe_register(str(name), handler, description, parameters)
        return registered

    # 2) Dict-like
    if isinstance(ext_obj, dict):
        for name, spec in ext_obj.items():
            handler = spec
            description = ""
            parameters = {"type": "object", "properties": {}, "required": []}
            if not callable(spec) and isinstance(spec, dict):
                handler = spec.get("handler") or spec.get("fn") or spec.get("callable")
                description = spec.get("description", "") or ""
                parameters = spec.get("parameters") or parameters
            _maybe_register(str(name), handler, description, parameters)
        return registered

    # 3) Iterable of callables
    try:
        iter(ext_obj)
        for fn in ext_obj:
            if not callable(fn):
                continue
            name = getattr(fn, "__tool_name__", None) or getattr(fn, "__name__", None) or "tool"
            description = getattr(fn, "__doc__", "") or ""
            _maybe_register(str(name), fn, description, {"type": "object", "properties": {}, "required": []})
        return registered
    except TypeError:
        pass

    logger.info("External tools object had unknown shape; nothing registered.")
    return registered


def _float_env(name: str, default: Optional[float]) -> Optional[float]:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return default
    try:
        return float(v)
    except Exception:
        return default


def _bool_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _introspect_model_from_obj(obj: Any) -> Optional[str]:
    """Try very hard to discover a model name on an arbitrary object."""
    if obj is None:
        return None

    # Common attribute names
    for attr in ("model_name", "model", "model_id", "_model_name", "_model", "_model_id"):
        val = getattr(obj, attr, None)
        if isinstance(val, str) and val.strip():
            return val.strip()

    # Callable getters
    for meth in ("get_model_name", "get_model", "model_name", "model"):
        if hasattr(obj, meth) and callable(getattr(obj, meth)):
            try:
                val = getattr(obj, meth)()
                if isinstance(val, str) and val.strip():
                    return val.strip()
            except Exception:
                pass

    # Nested config dict/obj
    cfg = getattr(obj, "config", None)
    if cfg is not None:
        if isinstance(cfg, dict):
            for k in ("model_name", "model", "model_id"):
                val = cfg.get(k)
                if isinstance(val, str) and val.strip():
                    return val.strip()
        else:
            for attr in ("model_name", "model", "model_id"):
                val = getattr(cfg, attr, None)
                if isinstance(val, str) and val.strip():
                    return val.strip()

    # Nested .client one level down
    inner = getattr(obj, "client", None)
    if inner is not None and inner is not obj:
        got = _introspect_model_from_obj(inner)
        if got:
            return got

    return None


def _env_model_hint() -> Optional[str]:
    """Last-ditch model hint from environment variables commonly used in your stack."""
    for k in (
        "GOOGLE_MODEL", "OPENAI_MODEL", "ANTHROPIC_MODEL",
        "LLM_MODEL", "MODEL_NAME", "MODEL",
    ):
        v = os.getenv(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


class AssistantAgent:
    """
    High-level orchestrator:
      - Owns the ADK client (initialized with centralized system prompt)
      - Loads tools via the agent_tools registry (aligned to current Mem0 API)
      - Optional mem0 search/save (per-call flags)
      - Optional RAG retrieval to prepend reference notes
      - Precise timing logs + optional LLM timeout guard
      - Explicit LLM call logging (start/end, sizes, previews)
    """

    def __init__(
        self,
        *,
        model_name: Optional[str] = None,
        instruction: Optional[str] = None,
        enable_rag: Optional[bool] = None,
        rag_top_k: int = 3,
        persona: str = "secretary",
        verbosity: str = "normal",
        # performance/observability controls:
        llm_timeout_s: Optional[float] = None,       # hard timeout for model call (None = no timeout)
        slow_llm_warn_s: float = 20.0,               # warn if model call exceeds this
        slow_rag_warn_s: float = 3.0,                # warn if RAG build exceeds this
        slow_mem0_warn_s: float = 1.0,               # warn if mem0 search exceeds this
    ) -> None:
        """
        Args:
            model_name: LLM id for ADKClient (e.g. "google/gemini-2.5-flash-preview-09-2025").
            instruction: If provided, overrides persona/verbosity prompt.
            enable_rag: Override RAG enablement (True/False). If None, falls back to config.
            rag_top_k: How many RAG notes to fetch when enabled.
            persona: One of {"secretary","executive","tutor"} for tone/role.
            verbosity: One of {"brief","normal","thorough"}.
            llm_timeout_s: Optional hard timeout for the LLM call via asyncio.wait_for.
            slow_*_warn_s: Emit a warning log if that stage takes longer than this many seconds.
        """
        # Allow environment overrides without changing call sites
        self._llm_timeout_s: Optional[float] = _float_env("LLM_TIMEOUT_S", llm_timeout_s)
        self._slow_llm_warn_s: float = _float_env("SLOW_LLM_WARN_S", slow_llm_warn_s) or slow_llm_warn_s
        self._slow_rag_warn_s: float = _float_env("SLOW_RAG_WARN_S", slow_rag_warn_s) or slow_rag_warn_s
        self._slow_mem0_warn_s: float = _float_env("SLOW_MEM0_WARN_S", slow_mem0_warn_s) or slow_mem0_warn_s

        # Optional payload previews in logs (disabled by default)
        self._log_llm_payload: bool = _bool_env("LLM_LOG_PAYLOAD", False)
        self._log_llm_max_chars: int = int(os.getenv("LLM_LOG_MAX_CHARS", "800"))

        # Mem0 wrapper (async) used for hydrate/save
        self.memory = Memory()

        # System prompt
        sys_inst = instruction or get_system_prompt(persona_key=persona, verbosity=verbosity)

        # ADK client
        self.client = ADKClient(model_name=model_name, instruction=sys_inst)

        # Resolve a reliable model name for logging up-front
        self._model_for_logs: str = (
            (model_name or "")  # explicit arg wins
            or (_introspect_model_from_obj(self.client) or "")
            or (_env_model_hint() or "")
            or "unknown"
        )
        if self._model_for_logs == "unknown":
            logger.debug("[llm] could not introspect model name; logging as 'unknown'")

        # ---- Tools: use the unified registry (agent_tools) ----
        registry = get_tool_registry()

        # Register first-party tools with the client
        try:
            # Handle both (registry) and (client, registry) signatures safely
            sig = inspect.signature(register_with_agent)
            params = list(sig.parameters.values())
            if len(params) == 1:
                register_with_agent(registry)          # type: ignore[arg-type]
            elif len(params) == 2:
                # Heuristic: if the first param is named like client/adk/agent, pass client first.
                first = params[0].name.lower()
                if any(k in first for k in ("client", "adk", "agent")):
                    register_with_agent(self.client, registry)  # type: ignore[arg-type]
                else:
                    register_with_agent(registry, self.client)  # type: ignore[arg-type]
            else:
                # Unknown signature -> fall back to manual registration
                for t in getattr(registry, "tools", []):
                    name = getattr(t, "name", None) or "tool"
                    handler = getattr(t, "handler", None) or None
                    if not callable(handler):
                        continue
                    description = getattr(t, "description", None) or getattr(handler, "__doc__", "") or ""
                    parameters = getattr(t, "parameters", None) or {"type": "object", "properties": {}, "required": []}
                    _register_tool_with_client(self.client, name, handler, description, parameters)
        except Exception as e:
            logger.warning(f"Failed to register tool registry: {e}")
            # Last resort: manual registration of first-party tools
            try:
                for t in getattr(registry, "tools", []):
                    name = getattr(t, "name", None) or "tool"
                    handler = getattr(t, "handler", None) or None
                    if not callable(handler):
                        continue
                    description = getattr(t, "description", None) or getattr(handler, "__doc__", "") or ""
                    parameters = getattr(t, "parameters", None) or {"type": "object", "properties": {}, "required": []}
                    _register_tool_with_client(self.client, name, handler, description, parameters)
            except Exception as e2:
                logger.warning(f"Manual registration of first-party tools also failed: {e2}")

        # ---- External toolpacks (register directly; filter duplicates) ----
        self._ext_tool_names: List[str] = []
        if load_external_tools:
            try:
                ext_obj = load_external_tools()
                # Build "used" set from first-party tools to avoid duplicate alias spam
                try:
                    first_party_names = [t.name for t in getattr(registry, "tools", [])]
                except Exception:
                    first_party_names = []
                used_norm_names = {_norm(n) for n in first_party_names} | set(_ADK_INTERNAL_ALIASES)

                self._ext_tool_names = _register_external_anyshape(self.client, ext_obj, used_norm_names)
                if self._ext_tool_names:
                    logger.info(f"Registered {len(self._ext_tool_names)} external tools with schema.")
            except Exception as e:
                logger.warning(f"No external tools found; continuing without plugins. {e}")

        # Track tool names for introspection
        try:
            first_party_names = [t.name for t in getattr(registry, "tools", [])]
        except Exception:
            first_party_names = []
        self._tool_names: List[str] = sorted(set(first_party_names + self._ext_tool_names))

        # ---- RAG toggle ----
        effective_rag = enable_rag if enable_rag is not None else DEFAULT_ENABLE_RAG
        self.rag = RAG(ChromaRetriever(CHROMA_DIR)) if (effective_rag and RAG and ChromaRetriever) else None
        self._rag_top_k = rag_top_k

    def list_tools(self) -> List[str]:
        """Return tool names registered on this agent."""
        return sorted(self._tool_names)

    async def _hydrate_memories(self, user_id: str, message: str, limit: int = 5) -> List[str]:
        # Uses your async wrapper in agents/assistant/memory.py
        return await self.memory.search(user_id, message, limit=limit)

    async def ainvoke(
        self,
        user_id: str,
        user_message: str,
        *,
        use_mem0_ctx: bool = False,
        save_mem0: bool = False,
        mem_k: int = 5,
    ) -> str:
        """
        Invoke the assistant for one turn.

        Args:
            user_id: Mem0 user id / session id for memory retrieval & save.
            user_message: The user's input.
            use_mem0_ctx: If True, search mem0 and include top-k memories in the user prompt.
            save_mem0: If True, persist Q/A to mem0 after answering.
            mem_k: Top-k memories to include when use_mem0_ctx=True.
        """
        logger.debug(f"[turn] uid={user_id} use_mem={use_mem0_ctx} save_mem={save_mem0} mem_k={mem_k}")
        t_all0 = time.perf_counter()

        # 1) mem0 memories (optional)
        memories: List[str] = []
        t0 = time.perf_counter()
        if use_mem0_ctx:
            try:
                memories = await self._hydrate_memories(user_id, user_message, limit=mem_k)
            except Exception as e:
                logger.warning(f"mem0 hydrate failed: {e}")
                memories = []
        t1 = time.perf_counter()
        dur_mem0 = t1 - t0
        if dur_mem0 > self._slow_mem0_warn_s:
            logger.warning(f"[perf] mem0 search slow: {dur_mem0:.2f}s (limit={mem_k})")
        else:
            logger.debug(f"[perf] mem0 search: {dur_mem0:.2f}s (items={len(memories)})")

        # 2) Build user content with optional memory block
        user_content = build_user_with_memories(user_message, memories=memories)

        # 3) Optionally append RAG notes
        notes_count = 0
        t2 = t1
        if self.rag:
            try:
                t_rag0 = time.perf_counter()
                notes = self.rag.gather_context(user_message, k=self._rag_top_k) or []
                t_rag1 = time.perf_counter()
                notes_count = len(notes)
                dur_rag = t_rag1 - t_rag0
                if dur_rag > self._slow_rag_warn_s:
                    logger.warning(f"[perf] RAG gather slow: {dur_rag:.2f}s (k={self._rag_top_k})")
                else:
                    logger.debug(f"[perf] RAG gather: {dur_rag:.2f}s (notes={notes_count})")
            except Exception as e:
                logger.warning(f"RAG gather_context failed: {e}")
                notes = []
            if notes:
                user_content = f"{user_content}\n\nReference notes:\n" + "\n".join(f"- {n}" for n in notes if n)
            t2 = time.perf_counter()

        # 4) Ask the ADK agent (LLM call) — with explicit LLM-call logs
        model_name = self._model_for_logs
        in_chars = len(user_content)

        if self._log_llm_payload:
            logger.info(f"[llm] → model={model_name} timeout={self._llm_timeout_s} "
                        f"mems={len(memories)} notes={notes_count} input_chars={in_chars}")
            logger.debug(f"[llm] prompt preview:\n{user_content[: self._log_llm_max_chars] + ('…' if len(user_content) > self._log_llm_max_chars else '')}")
        else:
            logger.info(f"[llm] → model={model_name} timeout={self._llm_timeout_s} "
                        f"mems={len(memories)} notes={notes_count} input_chars={in_chars}")

        t3 = t2
        try:
            if self._llm_timeout_s and self._llm_timeout_s > 0:
                reply = await asyncio.wait_for(self.client.run(user_content), timeout=self._llm_timeout_s)
            else:
                reply = await self.client.run(user_content)
        except asyncio.TimeoutError:
            logger.error(f"[llm] ✖ timeout after {self._llm_timeout_s:.1f}s for model={model_name}")
            reply = "Sorry—my response took too long. Please try again, or ask me to be briefer."
        except Exception as e:
            logger.error(f"[llm] ✖ error for model={model_name}: {e}")
            raise
        t4 = time.perf_counter()

        out_chars = len(reply or "")
        dur_llm = t4 - t3
        if self._log_llm_payload:
            logger.info(f"[llm] ← model={model_name} duration={dur_llm:.2f}s output_chars={out_chars}")
            logger.debug(f"[llm] response preview:\n{(reply or '')[: self._log_llm_max_chars] + ('…' if out_chars > self._log_llm_max_chars else '')}")
        else:
            logger.info(f"[llm] ← model={model_name} duration={dur_llm:.2f}s output_chars={out_chars}")

        if dur_llm > self._slow_llm_warn_s:
            logger.warning(f"[perf] LLM call slow: {dur_llm:.2f}s (model may be cold/queued)")
        else:
            logger.debug(f"[perf] LLM call: {dur_llm:.2f}s")

        # 5) Persist to mem0 (optional; fast)
        if save_mem0:
            t5 = time.perf_counter()
            try:
                await self.memory.add_user_text(user_id, user_message)
                await self.memory.add_agent_text(user_id, reply)
            except Exception as e:
                logger.warning(f"mem0 save failed: {e}")
            t6 = time.perf_counter()
            logger.debug(f"[perf] mem0 save: {(t6 - t5):.2f}s")

        logger.debug(f"[perf] end-to-end: {(time.perf_counter() - t_all0):.2f}s")
        return reply
