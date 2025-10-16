# agents/assistant/agent.py

from __future__ import annotations
from typing import List, Dict, Optional
from loguru import logger

from .memory import Memory
from .adk_client import ADKClient
from .prompts.prompt import get_system_prompt, build_user_with_memories
from .tools.builtin_tools import builtin_tools
from .tools.external_loader import load_external_tools

# RAG is optional; guard imports so the agent still works without it
try:
    from .config import ENABLE_RAG as DEFAULT_ENABLE_RAG, CHROMA_DIR
    from .retrievers.chroma_retriever import ChromaRetriever
    from .rag import RAG
except Exception:
    DEFAULT_ENABLE_RAG = False  # type: ignore[assignment]
    CHROMA_DIR = ""             # type: ignore[assignment]
    RAG = None                  # type: ignore[assignment]
    ChromaRetriever = None      # type: ignore[assignment]


class AssistantAgent:
    """
    High-level orchestrator:
      - Owns the ADK client (initialized with centralized system prompt)
      - Loads tools (builtin + external) and registers them
      - Optional mem0 search/save
      - Optional RAG retrieval to prepend reference notes
    """

    def __init__(
        self,
        *,
        model_name: Optional[str] = None,
        instruction: Optional[str] = None,
        enable_rag: Optional[bool] = None,
        rag_top_k: int = 3,
    ) -> None:
        """
        Args:
            model_name: LLM id for ADKClient (e.g. "google/gemini-2.5-flash-preview-09-2025").
            instruction: System prompt; defaults to prompts.prompt.get_system_prompt().
            enable_rag: Override RAG enablement (True/False). If None, falls back to config.
            rag_top_k: How many RAG notes to fetch when enabled.
        """
        self.memory = Memory()

        # System instruction (your centralized prompt)
        sys_inst = instruction or get_system_prompt()

        # ADK client
        self.client = ADKClient(model_name=model_name, instruction=sys_inst)

        # Register tools (builtin + external). register_tool(fn) handles naming internally.
        tools: Dict[str, callable] = {}
        try:
            tools.update(builtin_tools())
        except Exception as e:
            logger.warning(f"builtin tools unavailable: {e}")
        try:
            tools.update(load_external_tools())
        except Exception as e:
            logger.warning(f"external tools unavailable: {e}")

        for fn in tools.values():
            # ADKClient handles naming and duplicate protection internally
            self.client.register_tool(fn)

        # RAG toggle: param overrides config; requires RAG + retriever available
        effective_rag = enable_rag if enable_rag is not None else DEFAULT_ENABLE_RAG
        self.rag = RAG(ChromaRetriever(CHROMA_DIR)) if (effective_rag and RAG and ChromaRetriever) else None
        self._rag_top_k = rag_top_k

    async def _hydrate_memories(self, user_id: str, message: str, limit: int = 5) -> List[str]:
        return await self.memory.search(user_id, message, limit=limit)

    async def ainvoke(self, user_id: str, user_message: str) -> str:
        # 1) mem0 memories
        memories = await self._hydrate_memories(user_id, user_message)

        # 2) Your helper (only supports 'memories')
        user_content = build_user_with_memories(user_message, memories=memories)

        # 3) Optionally append RAG notes (without changing your prompt API)
        if self.rag:
            try:
                notes = self.rag.gather_context(user_message, k=self._rag_top_k) or []
            except Exception as e:
                logger.warning(f"RAG gather_context failed: {e}")
                notes = []
            if notes:
                user_content = f"{user_content}\n\nReference notes:\n" + "\n".join(f"- {n}" for n in notes if n)

        # 4) Ask the ADK agent
        reply = await self.client.run(user_content)

        # 5) Persist to mem0
        await self.memory.add_user_text(user_id, user_message)
        await self.memory.add_agent_text(user_id, reply)

        return reply
