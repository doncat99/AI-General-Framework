# agents/assistant/config.py
from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()

APP_NAME = os.getenv("APP_NAME", "adk_mem0_assistant")
DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID", "default_user")

# Vector store (optional RAG)
ENABLE_RAG = os.getenv("ENABLE_RAG", "1") == "1"
CHROMA_DIR = os.getenv("CHROMA_DIR", ".chroma")
