# agents/assistant/agent.py
from __future__ import annotations

import os
import time
import asyncio
from typing import Optional, List, Dict, Any, Tuple

import httpx
from loguru import logger

from workshop.assistant.adk_tool_client import AdkToolClient
from workshop.assistant.tools.tool_loader import register_tools
from workshop.assistant.prompts.prompt import get_system_prompt

# Share the ContextVar used for per-turn correlation in tool logs
try:
    from workshop.assistant.adk_tool_client import TURN_ID
except Exception:
    TURN_ID = None


def _build_user_with_memories(user_message: str, memories: List[str]) -> str:
    if not memories:
        return user_message
    mem_block = "\n".join(f"- {m}" for m in memories if m)
    return f"{user_message}\n\n[Related memories]\n{mem_block}"


def _build_user_with_notes(content: str, notes: List[str]) -> str:
    if not notes:
        return content
    note_block = "\n".join(f"- {n}" for n in notes if n)
    return f"{content}\n\n[Reference notes]\n{note_block}"


def _preview_list(items: List[str], max_items: int = 5, max_chars: int = 200) -> str:
    out = []
    for i, s in enumerate(items[:max_items], 1):
        s = s if len(s) <= max_chars else s[:max_chars] + "…"
        out.append(f"{i}. {s}")
    return "\n".join(out)


class AssistantAgent:
    """
    Orchestrator:
      - Central system prompt + ADK client
      - Tool loading
      - Optional mem0 search/save + optional RAG via mem0 index
      - Detailed timing with a 'time_consume' summary stored per turn
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
        # performance/observability:
        llm_timeout_s: Optional[float] = None,
        slow_llm_warn_s: float = 20.0,
        slow_rag_warn_s: float = 3.0,
        slow_mem0_warn_s: float = 1.0,
        # mem0 overrides:
        mem0_base_url: Optional[str] = None,
        mem0_api_key: Optional[str] = None,
    ) -> None:
        # perf controls (env overrides allowed)
        self._llm_timeout_s = self._parse_float_env("LLM_TIMEOUT_S", llm_timeout_s)
        self._slow_llm_warn_s = self._parse_float_env("SLOW_LLM_WARN_S", slow_llm_warn_s) or slow_llm_warn_s
        self._slow_rag_warn_s = self._parse_float_env("SLOW_RAG_WARN_S", slow_rag_warn_s) or slow_rag_warn_s
        self._slow_mem0_warn_s = self._parse_float_env("SLOW_MEM0_WARN_S", slow_mem0_warn_s) or slow_mem0_warn_s

        # logging knobs
        self._log_llm_payload = self._parse_bool_env("LLM_LOG_PAYLOAD", False)
        self._log_llm_max_chars = int(os.getenv("LLM_LOG_MAX_CHARS", "800"))

        # mem0 base URL + optional auth
        self._mem0_url: str = (mem0_base_url or self._resolve_mem0_url()).rstrip("/")
        self._mem0_api_key: Optional[str] = (
            mem0_api_key
            or os.getenv("MEM0_SERVICE_API_KEY")
            or os.getenv("MEM0_API_KEY")
            or None
        )

        # system prompt + model
        sys_inst = instruction or get_system_prompt(persona_key=persona, verbosity=verbosity)

        # ---- ADK client ----
        self.client = AdkToolClient(model_name=model_name, instruction=sys_inst)

        # ---- register tool packs (mem0 pack, etc.) ----
        register_tools(self.client)

        # ---- model name for logs (best-effort) ----
        self._model_for_logs: str = model_name or "unknown"

        # ---- RAG toggle ----
        if enable_rag is None:
            enable_rag = self._parse_bool_env("ENABLE_RAG", True)
        self._rag_enabled = bool(enable_rag)
        self._rag_top_k = int(rag_top_k)

        # per-turn correlation & timing
        self._turn_counter = 0
        self._last_time_consume: Dict[str, Any] = {}

    # ---------- public: last timing ----------
    def last_time_consume(self) -> Dict[str, Any]:
        """Return timing summary for the most recent turn."""
        return dict(self._last_time_consume)

    # ---------- env helpers ----------
    @staticmethod
    def _parse_bool_env(key: str, default: bool) -> bool:
        raw = os.getenv(key, None)
        if raw is None:
            return default
        v = raw.strip().lower()
        if v in ("1", "true", "yes", "y", "on"):
            return True
        if v in ("0", "false", "no", "n", "off"):
            return False
        return default

    @staticmethod
    def _parse_float_env(key: str, default: Optional[float]) -> Optional[float]:
        raw = os.getenv(key, None)
        if raw is None or raw == "":
            return default
        try:
            return float(raw)
        except Exception:
            return default

    @staticmethod
    def _inside_container() -> bool:
        return bool(os.getenv("INSIDE_DOCKER") or os.getenv("KUBERNETES_SERVICE_HOST"))

    @classmethod
    def _resolve_mem0_url(cls) -> str:
        # 1) explicit override via env
        env_url = os.getenv("MEM0_SERVER_URL")
        if env_url:
            return env_url
        # 2) if inside container, prefer service DNS
        if cls._inside_container():
            return "http://mem0:8000"
        # 3) host default uses published port (default 8001)
        port = os.getenv("MEM0_PORT", "8001")
        return f"http://localhost:{port}"

    def _mem0_headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self._mem0_api_key:
            h["Authorization"] = f"Bearer {self._mem0_api_key}"
        return h

    def _mem0_client(self, *, timeout: float = 20.0) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self._mem0_url, headers=self._mem0_headers(), timeout=timeout)

    # ---------- mem0 helpers ----------
    async def _mem0_add(self, user_id: str, text: str, role: str) -> None:
        """
        Persist a single message into mem0 for given user_id.
        role: "user" or "assistant" (stored in metadata)
        """
        payload = {
            "message": text,
            "user_id": user_id,
            "agent_id": "rag_agent",
            "metadata": {"role": role},
        }
        async with self._mem0_client(timeout=20.0) as cli:
            r = await cli.post("/add", json=payload)
            r.raise_for_status()

    async def _mem0_search(self, user_id: str, query: str, limit: int) -> List[str]:
        """
        Search mem0 semantic memory (not index).
        Returns list of memory texts.
        """
        payload = {
            "query": query,
            "user_id": user_id,
            "agent_id": "rag_agent",
            "limit": int(limit),
        }
        async with self._mem0_client(timeout=20.0) as cli:
            r = await cli.post("/search", json=payload)
            r.raise_for_status()
            data = r.json() or {}
            items = data.get("memories") if isinstance(data, dict) else None
            if not items:
                # Some gateways return {"results":[{"memory":...}, ...]}
                items = (data.get("results") if isinstance(data, dict) else []) or []
            out: List[str] = []
            for m in items:
                if isinstance(m, str):
                    out.append(m)
                elif isinstance(m, dict):
                    out.append(m.get("text") or m.get("memory") or m.get("content") or "")
            return [s for s in out if s]

    async def _mem0_index_search(self, query: str, k: int) -> List[str]:
        """
        Query Chroma index via mem0 server’s /index/search.
        Returns list of document strings.
        """
        payload = {"query": query, "k": int(k), "where": {}}
        async with self._mem0_client(timeout=20.0) as cli:
            r = await cli.post("/index/search", json=payload)
            r.raise_for_status()
            data = r.json() or {}
            # The server returns {"documents": [...]}; some variants may return {"hits":[{"page_content":...}]}
            docs = data.get("documents") or []
            if docs:
                return [d for d in docs if isinstance(d, str) and d]
            hits = data.get("hits") or []
            out: List[str] = []
            for h in hits:
                if isinstance(h, dict):
                    txt = h.get("page_content") or h.get("text") or h.get("document")
                    if isinstance(txt, str) and txt:
                        out.append(txt)
            return out

    # ---------- main entry ----------
    async def ainvoke(
        self,
        user_id: str,
        user_message: str,
        *,
        use_mem0_ctx: bool = False,
        save_mem0: bool = False,
        mem_k: int = 5,
        system_prompt: Optional[str] = None,
    ) -> str:
        # per-turn correlation id
        self._turn_counter += 1
        if TURN_ID:
            TURN_ID.set(str(self._turn_counter))
        logger.info(f"[turn {self._turn_counter}] user='{user_id}'")

        # reset last timings
        self._last_time_consume = {}
        t_all0 = time.perf_counter()

        # 1) fetch memories + notes concurrently
        async def _get_memories() -> Tuple[List[str], float]:
            if not use_mem0_ctx:
                return [], 0.0
            t0 = time.perf_counter()
            try:
                res = await self._mem0_search(user_id, user_message, limit=mem_k)
            except Exception as e:
                logger.warning(f"mem0 hydrate failed: {e}")
                res = []
            return res, time.perf_counter() - t0

        async def _get_notes() -> Tuple[List[str], float]:
            if not self._rag_enabled or self._rag_top_k <= 0:
                return [], 0.0
            t0 = time.perf_counter()
            try:
                res = await self._mem0_index_search(user_message, k=self._rag_top_k)
            except Exception as e:
                logger.warning(f"RAG gather_context failed: {e}")
                res = []
            return res, time.perf_counter() - t0

        (memories, dur_mem0), (notes, dur_rag) = await asyncio.gather(_get_memories(), _get_notes())

        # perf logs
        if use_mem0_ctx:
            if dur_mem0 > self._slow_mem0_warn_s:
                logger.warning(f"[perf] mem0 search slow: {dur_mem0:.2f}s (limit={mem_k})")
            else:
                logger.debug(f"[perf] mem0 search: {dur_mem0:.2f}s (items={len(memories)})")
            if memories:
                logger.debug("[ctx] memories preview:\n" + _preview_list(memories))
        if self._rag_enabled and self._rag_top_k > 0:
            if dur_rag > self._slow_rag_warn_s:
                logger.warning(f"[perf] RAG gather slow: {dur_rag:.2f}s (k={self._rag_top_k})")
            else:
                logger.debug(f"[perf] RAG gather: {dur_rag:.2f}s (notes={len(notes)})")
            if notes:
                logger.debug("[ctx] notes preview:\n" + _preview_list(notes))

        # 2) build prompt
        content = _build_user_with_memories(user_message, memories)
        content = _build_user_with_notes(content, notes)

        # 3) LLM call
        model_name = self._model_for_logs
        in_chars = len(content)
        if self._log_llm_payload:
            logger.info(
                f"[llm] → model={model_name} timeout={self._llm_timeout_s} "
                f"mems={len(memories)} notes={len(notes)} input_chars={in_chars}"
            )
            preview = content[: self._log_llm_max_chars]
            if len(content) > self._log_llm_max_chars:
                preview += "…"
            logger.debug(f"[llm] prompt preview:\n{preview}")
        else:
            logger.info(
                f"[llm] → model={model_name} timeout={self._llm_timeout_s} "
                f"mems={len(memories)} notes={len(notes)} input_chars={in_chars}"
            )

        t_llm0 = time.perf_counter()
        try:
            if self._llm_timeout_s and self._llm_timeout_s > 0:
                reply = await asyncio.wait_for(
                    self.client.run_async(content, system_prompt=system_prompt),
                    timeout=self._llm_timeout_s,
                )
            else:
                reply = await self.client.run_async(content, system_prompt=system_prompt)
        except asyncio.TimeoutError:
            logger.error(f"[llm] ✖ timeout after {self._llm_timeout_s:.1f}s for model={model_name}")
            reply = "Sorry—my response took too long. Please try again, or ask me to be briefer."
        except Exception as e:
            logger.error(f"[llm] ✖ error for model={model_name}: {e}")
            raise
        dur_llm = time.perf_counter() - t_llm0

        out_chars = len(reply or "")
        if self._log_llm_payload:
            logger.info(f"[llm] ← model={model_name} duration={dur_llm:.2f}s output_chars={out_chars}")
            preview = (reply or "")[: self._log_llm_max_chars]
            if out_chars > self._log_llm_max_chars:
                preview += "…"
            logger.debug(f"[llm] response preview:\n{preview}")
        else:
            logger.info(f"[llm] ← model={model_name} duration={dur_llm:.2f}s output_chars={out_chars}")

        if dur_llm > self._slow_llm_warn_s:
            logger.warning(f"[perf] LLM call slow: {dur_llm:.2f}s (model may be cold/queued)")
        else:
            logger.debug(f"[perf] LLM call: {dur_llm:.2f}s")

        # 4) persist to mem0 (optional)
        dur_save = 0.0
        if save_mem0:
            t5 = time.perf_counter()
            try:
                await self._mem0_add(user_id, user_message, role="user")
                await self._mem0_add(user_id, reply or "", role="assistant")
            except Exception as e:
                logger.warning(f"mem0 save failed: {e}")
            dur_save = time.perf_counter() - t5
            logger.debug(f"[perf] mem0 save: {dur_save:.2f}s")

        total_s = time.perf_counter() - t_all0

        # ---- record + emit time_consume summary ----
        self._last_time_consume = {
            "model": model_name,
            "mem0_search_s": round(dur_mem0, 4),
            "rag_s": round(dur_rag, 4),
            "llm_s": round(dur_llm, 4),
            "save_s": round(dur_save, 4),
            "total_s": round(total_s, 4),
            "memories": len(memories),
            "notes": len(notes),
            "input_chars": in_chars,
            "output_chars": out_chars,
        }
        logger.info(
            "[time_consume] "
            f"mem0={self._last_time_consume['mem0_search_s']}s "
            f"rag={self._last_time_consume['rag_s']}s "
            f"llm={self._last_time_consume['llm_s']}s "
            f"save={self._last_time_consume['save_s']}s "
            f"total={self._last_time_consume['total_s']}s "
            f"(mems={len(memories)} notes={len(notes)})"
        )

        logger.debug(f"[perf] end-to-end: {total_s:.2f}s")
        return reply or ""

    # Handy for dashboards / tests
    def list_tools(self) -> List[str]:
        try:
            return sorted([getattr(t, "__name__", "tool") for t in self.client.tools])
        except Exception:
            return []
