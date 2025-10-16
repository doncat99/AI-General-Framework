# agents/assistant_adk/prompts.py
from __future__ import annotations
from typing import List

def render_system_prompt() -> str:
    return (
        "You are a helpful AI assistant. Use retrieved notes/memories when relevant "
        "and cite them in natural language (no URLs). Be concise, precise, actionable."
    )

def render_user_prompt(user_message: str, memories: List[str], rag_notes: List[str]) -> str:
    mem_block = ""
    if memories:
        mem_block = "Relevant memories:\n" + "\n".join(f"- {m}" for m in memories) + "\n\n"
    rag_block = ""
    if rag_notes:
        rag_block = "Reference notes:\n" + "\n".join(f"- {n}" for n in rag_notes) + "\n\n"
    return f"{mem_block}{rag_block}User: {user_message}"
