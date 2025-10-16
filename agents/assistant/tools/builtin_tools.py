# agents/assistant/tools/builtin_tools.py
from __future__ import annotations
from typing import Dict, Any
from loguru import logger

# --- Lightweight, always-safe tool(s) ---
async def echo(text: str) -> str:
    """Return the given text prefixed by ECHO (smoke-test tool)."""
    return f"ECHO: {text}"

# --- Heavier tool(s) with LAZY imports to avoid config side-effects ---
async def ingest_roadmap(json_path: str) -> str:
    """
    Ingest a JSON roadmap into mem0.
    Imports the ingest pipeline only when this tool is actually invoked,
    so 'assistant tools' and other lightweight commands don't pull config.
    """
    try:
        # Lazy imports here to avoid triggering settings at module import
        from agents.assistant.ingest.pipeline import IngestPipeline
        from agents.assistant.ingest.sources.json_roadmap import JSONRoadmapSource
        from agents.assistant.ingest.sinks.mem0_sink import Mem0Sink
    except Exception as e:
        logger.error(f"Failed to import ingest stack: {e}")
        return f"Failed to import ingest stack: {e}"

    try:
        pipe = IngestPipeline(JSONRoadmapSource(json_path), sinks=[Mem0Sink()])
        pipe.run()
        return f"Ingested roadmap into mem0 from {json_path}"
    except Exception as e:
        logger.error(f"Ingest failed: {e}")
        return f"Ingest failed: {e}"

def builtin_tools() -> Dict[str, Any]:
    """
    Return a mapping of tool name -> async callable.
    Keep this module import-time light (no heavy imports at top-level).
    """
    return {
        "echo": echo,
        "ingest_roadmap": ingest_roadmap,
    }
