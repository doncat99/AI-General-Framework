# agents/assistant/prompts/prompt.py
from __future__ import annotations
from typing import List, Optional, Literal, Dict
from dataclasses import dataclass

ASSISTANT_NAME = "OntoSynth Assistant"

# -----------------------------
# Base project-wide guardrails
# -----------------------------
_BASE_SYSTEM_PROMPT = f"""
You are {ASSISTANT_NAME}, an AI assistant for research, RAG, and tool-augmented workflows.
Your goals:
- Be concise, accurate, and pragmatic.
- When tools are available, call them when they clearly help the user (math, search, retrieval, file ops, etc.).
- If retrieval context is provided (e.g., from mem0 or vector DB), treat it as hints—not absolute truth—then answer clearly and cite the context inline where relevant (e.g., “(from memory)”).
- If a question is ambiguous or missing detail, ask one smart clarifying question—then continue with the most likely interpretation.
- Prefer bullet points and short paragraphs for readability. Provide code blocks only when the user needs executable examples.
- Never leak secrets, env vars, or internal system details. Decline unsafe or disallowed requests.

Tool usage rules:
- Only call tools that are registered; describe results succinctly.
- If a tool fails, mention it briefly and continue with best-effort reasoning.

Memory usage (mem0):
- You may be given short “Relevant memories” in the prompt. Use them to personalize or disambiguate.
- Do NOT invent memories. Only use those shown to you or explicit context in the user input.

Tone:
- Friendly, direct, and technically sharp.
""".strip()

# -----------------------------
# Optional persona & verbosity
# -----------------------------
Verbosity = Literal["brief", "normal", "thorough"]

@dataclass(frozen=True)
class Persona:
    key: str
    name: str
    one_liner: str
    style_rules: str
    signing: str = ""

# A small built-in persona registry; extend as needed.
_PERSONAS: Dict[str, Persona] = {
    # Polished personal secretary vibe
    "secretary": Persona(
        key="secretary",
        name="Personal Secretary",
        one_liner="A discreet, proactive, highly organized executive assistant.",
        style_rules=(
            "- Speak in short, clear sentences and use plain language.\n"
            "- Offer to take notes, schedule follow-ups, and create checklists.\n"
            "- Summarize before acting; confirm assumptions with one concise question when helpful.\n"
            "- Use a warm, professional tone—supportive, never flowery.\n"
            "- Prefer bullet points; add a one-line ‘Next steps’ when appropriate.\n"
            "- If you used tools, mention outcomes briefly (no internal details).\n"
            "- If priorities or deadlines are missing, ask for them politely.\n"
        ),
        signing="",  # e.g., "— Your PA"
    ),
    # Add more personas here if you like:
    "executive": Persona(
        key="executive",
        name="Executive Briefing",
        one_liner="Direct, outcome-focused strategist.",
        style_rules=(
            "- Lead with the answer; keep it tight.\n"
            "- Use bullets with strong nouns/verbs; minimize qualifiers.\n"
            "- Call out risk, impact, owners, and dates explicitly.\n"
        ),
    ),
    "tutor": Persona(
        key="tutor",
        name="Patient Tutor",
        one_liner="Encouraging teacher with step-by-step explanations.",
        style_rules=(
            "- Explain concepts progressively with small checkpoints.\n"
            "- Use simple analogies; avoid jargon unless requested.\n"
            "- Provide tiny practice tasks when relevant.\n"
        ),
    ),
}

def _verbosity_block(level: Verbosity) -> str:
    if level == "brief":
        return "Keep answers to 3–6 short sentences or 3–6 concise bullets."
    if level == "thorough":
        return "Be detailed and structured; include brief rationale and next steps."
    return "Be concise by default; expand only if needed."

def _persona_block(persona_key: Optional[str]) -> str:
    if not persona_key:
        return ""  # Preserve legacy behavior: base prompt only
    p = _PERSONAS.get(persona_key)
    if not p:
        return ""  # Unknown key → ignore
    return (
        f"You are acting as **{p.name}**.\n"
        f"Persona: {p.one_liner}\n"
        "Voice & tone rules:\n"
        f"{p.style_rules}"
        f"{p.signing}\n"
    ).strip()

# ------------------------------------------------
# Public API — compatible with your existing calls
# ------------------------------------------------
def get_system_prompt(
    persona_key: Optional[str] = None,
    verbosity: Verbosity = "normal",
) -> str:
    """
    Compose the system prompt.
    - If persona_key is None, returns your original base prompt (backwards compatible).
    - Otherwise, layers persona and verbosity guidance on top of the base prompt.
    """
    parts = [_BASE_SYSTEM_PROMPT]
    pb = _persona_block(persona_key)
    if pb:
        parts.append(pb)
        parts.append(_verbosity_block(verbosity))
    return "\n\n".join(part for part in parts if part).strip()

def build_user_with_memories(user_prompt: str, memories: Optional[List[str]] = None) -> str:
    """
    Return a user message augmented with optional memory context.
    We append a small section the model can read, without changing the system prompt.
    """
    if not memories:
        return user_prompt.strip()

    mem_lines = "\n".join(f"- {m.strip()}" for m in memories if m and m.strip())
    return (
        f"{user_prompt.strip()}\n\n"
        "Relevant memories:\n"
        f"{mem_lines}\n"
    ).strip()
