# agents/assistant/agent.py
from __future__ import annotations
from typing import List, Dict
from loguru import logger

from .config import ENABLE_RAG, CHROMA_DIR
from .memory import Memory
from .adk_client import ADKClient
from .prompts import render_system_prompt, render_user_prompt
from .rag import RAG
from .retrievers.chroma_retriever import ChromaRetriever
from .tools.builtin_tools import builtin_tools
from .tools.external_loader import load_external_tools

class AssistantAgent:
    def __init__(self) -> None:
        self.memory = Memory()
        self.client = ADKClient()

        # Register tools (builtin + external)
        for name, fn in {**builtin_tools(), **load_external_tools()}.items():
            self.client.register_tool(name, fn)

        self.rag = RAG(ChromaRetriever(CHROMA_DIR)) if ENABLE_RAG else None

    async def _hydrate_memories(self, user_id: str, message: str) -> List[str]:
        raw = await self.memory.search(user_id, message, limit=5)
        return raw

    async def ainvoke(self, user_id: str, user_message: str) -> str:
        memories = await self._hydrate_memories(user_id, user_message)
        rag_notes: List[str] = []
        if self.rag:
            rag_notes = self.rag.gather_context(user_message, k=3)

        system = render_system_prompt()
        user = render_user_prompt(user_message, memories, rag_notes)
        full_prompt = f"{system}\n\n{user}"

        reply = await self.client.run(full_prompt)

        # persist conversation into mem0
        await self.memory.add_user_text(user_id, user_message)
        await self.memory.add_agent_text(user_id, reply)
        return reply
