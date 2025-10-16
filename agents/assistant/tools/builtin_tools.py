# agents/assistant/tools/builtin_tools.py
from __future__ import annotations
from typing import Dict, Any
from agents.assistant.ingest.pipeline import IngestPipeline
from agents.assistant.ingest.sources.json_roadmap import JSONRoadmapSource
from agents.assistant.ingest.sinks.mem0_sink import Mem0Sink

async def echo_tool(text: str) -> str:
    return f"ECHO: {text}"

def builtin_tools() -> Dict[str, Any]:
    return {
        "echo": echo_tool,
    }

async def ingest_roadmap_tool(json_path: str) -> str:
    pipe = IngestPipeline(JSONRoadmapSource(json_path), sinks=[Mem0Sink()])
    pipe.run()
    return f"Ingested roadmap into mem0 from {json_path}"

def builtin_tools() -> Dict[str, Any]:
    return {
        "echo": (lambda text: f"ECHO: {text}"),  # keep your echo if you like
        "ingest_roadmap": ingest_roadmap_tool,
    }
