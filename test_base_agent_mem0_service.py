# /Users/huangdon/Documents/OntoSynth/test_base_agent_mem0_service_fixed.py
# Python 3.13

import os
import json
import asyncio
import warnings
from textwrap import shorten
from typing import Optional, Dict, Any, List

import requests
from dotenv import load_dotenv
from loguru import logger

# ---- Your base agent (refined w/ loguru) ----
from utilities.base.base_agent import BaseAgent

# ---- Google ADK types (used by BaseAgent internally) ----
from google.genai import types  # noqa: F401  (import kept for completeness)

# ---- Langfuse (current Python SDK) ----
try:
    from langfuse import get_client as lf_get_client
except Exception:
    lf_get_client = None

warnings.filterwarnings("ignore", category=DeprecationWarning)
load_dotenv()

# ------------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------------
APP_NAME   = "healthcare_assistant_app"
USER_ID    = "Alex"
SESSION_ID = "session_001"

# OpenRouter-style model id; BaseAgent normalizes to `openrouter/<...>`
MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-5")

# Mem0 service (YOUR self-hosted server)
# Matches your compose: host exposes mem0 on 8001 -> http://127.0.0.1:8001
MEM0_BASE_URL = os.getenv("MEM0_SERVER_URL", "http://127.0.0.1:8001")
MEM0_API_KEY  = os.getenv("MEM0_API_KEY", "")  # optional, not required by your server

# Optional: quieter console
logger.remove()
logger.add(
    sink=lambda m: print(m, end=""),
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>\n",
)

# ------------------------------------------------------------------------------------
# Minimal Mem0 Service Client (matches YOUR FastAPI server)
# ------------------------------------------------------------------------------------
class Mem0ServiceClient:
    """
    Calls YOUR Mem0 server (gateway/mem0/service.py):

      POST {base}/add
        body: { "message": str, "user_id": str, "agent_id": str, "metadata": {...} }

      POST {base}/search
        body: { "query": str, "user_id": str, "agent_id": str, "limit": int }

    Returns harmless defaults if the server is unreachable, so the demo keeps running.
    """

    def __init__(self, base_url: str, api_key: str = "", timeout: float = 8.0, agent_id: str = "rag_agent"):
        self.base_url = base_url.rstrip("/")
        self.api_key  = api_key
        self.timeout  = timeout
        self.agent_id = agent_id
        self._local_fallback_store: Dict[str, List[Dict[str, Any]]] = {}  # {user_id: [{"message":..., "metadata":...}]}

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def add(self, *, message: str, user_id: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        url = f"{self.base_url}/add"
        payload = {"message": message, "user_id": user_id, "agent_id": self.agent_id, "metadata": metadata or {}}
        try:
            resp = requests.post(url, headers=self._headers(), json=payload, timeout=self.timeout)
            if resp.status_code == 200:
                return True
            logger.warning(f"[mem0] add non-200={resp.status_code} body={resp.text[:200]}")
        except Exception as e:
            logger.warning(f"[mem0] add HTTP error: {e}")

        # fallback to local store
        self._local_fallback_store.setdefault(user_id, []).append({"message": message, "metadata": metadata or {}})
        return True

    def search(self, *, query: str, user_id: str, limit: int = 5) -> List[str]:
        url = f"{self.base_url}/search"
        payload = {"query": query, "user_id": user_id, "agent_id": self.agent_id, "limit": limit}
        try:
            resp = requests.post(url, headers=self._headers(), json=payload, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                # Your server returns: {"status":"success","memories":[...],"count":N}
                items = data.get("memories", []) if isinstance(data, dict) else []
                out: List[str] = []
                for it in items:
                    if isinstance(it, str):
                        out.append(it)
                    elif isinstance(it, dict):
                        # Try common shapes
                        if it.get("memory"):
                            out.append(it["memory"])
                        elif it.get("text"):
                            out.append(it["text"])
                        elif it.get("message"):
                            out.append(it["message"])
                return out
            logger.warning(f"[mem0] search non-200={resp.status_code} body={resp.text[:200]}")
        except Exception as e:
            logger.warning(f"[mem0] search HTTP error: {e}")

        # fallback: naive local search
        out: List[str] = []
        for rec in self._local_fallback_store.get(user_id, []):
            msg = rec.get("message", "")
            if msg and any(w.lower() in msg.lower() for w in query.split()):
                out.append(msg)
            if len(out) >= limit:
                break
        return out


mem0_client = Mem0ServiceClient(MEM0_BASE_URL, MEM0_API_KEY)

# ------------------------------------------------------------------------------------
# Langfuse helper
# ------------------------------------------------------------------------------------
def get_langfuse():
    if lf_get_client is None:
        logger.info("[lf] Langfuse not active (package not installed)")
        return None
    try:
        client = lf_get_client()
        host = os.getenv("LANGFUSE_HOST", "(unset)")
        logger.info(f"[lf] Langfuse client initialized host={host}")
        return client
    except Exception as e:
        logger.info(f"[lf] Langfuse not active ({e})")
        return None

LF = get_langfuse()

# ------------------------------------------------------------------------------------
# Tools
# ------------------------------------------------------------------------------------
def save_patient_info(information: str) -> dict:
    uid = USER_ID
    short = shorten(information, 80, placeholder="...")
    logger.info(f"[tool] save_patient_info(uid={uid}) ← {short}")
    ok = mem0_client.add(message=information, user_id=uid, metadata={"type": "patient_information"})
    logger.info(f"[tool] save_patient_info → ok={ok}")
    return {"status": "success" if ok else "error"}

def retrieve_patient_info(query: str) -> dict:
    uid = USER_ID
    short = shorten(query, 60, placeholder="...")
    logger.info(f"[tool] retrieve_patient_info(uid={uid}) ← {short}")
    hits = mem0_client.search(query=query, user_id=uid, limit=5)
    logger.info(f"[tool] retrieve_patient_info → {len(hits)} hit(s)")
    return {"status": "success" if hits else "no_results", "memories": hits, "count": len(hits)}

def schedule_appointment(date: str, time: str, reason: str) -> dict:
    appt_id = f"APT-{abs(hash(date + time)) % 10000:04d}"
    logger.info(f"[tool] schedule_appointment ← date='{date}' time='{time}' reason='{reason}'")
    return {
        "status": "success",
        "appointment_id": appt_id,
        "confirmation": f"Appointment scheduled for {date} at {time} for {reason}",
        "message": "Please arrive 15 minutes early.",
    }

# ------------------------------------------------------------------------------------
# Build the agent using your BaseAgent
# ------------------------------------------------------------------------------------
INSTRUCTION = """
You are a helpful Healthcare Assistant with memory capabilities.

Primary responsibilities:
1) Remember patient information using 'save_patient_info' when they share symptoms, conditions, or preferences.
2) Retrieve past information using 'retrieve_patient_info' when relevant.
3) Help schedule appointments using 'schedule_appointment'.

Guidelines:
- Be empathetic and professional; you are not a doctor (no diagnosis/treatment).
- Save important info (symptoms, conditions, allergies, preferences).
- Before re-asking, check if relevant info exists.
- For serious symptoms, advise consulting a healthcare professional.
- Keep all patient info confidential.
""".strip()

agent = BaseAgent(
    agent_name="healthcare_assistant",
    model_name=MODEL,
    instruction=INSTRUCTION,
    tools=[save_patient_info, retrieve_patient_info, schedule_appointment],
    app_name=APP_NAME,
    user_id=USER_ID,
    session_id=SESSION_ID,
)

# ------------------------------------------------------------------------------------
# Conversation helper (Langfuse spans included if available)
# ------------------------------------------------------------------------------------
async def ask(prompt: str) -> str:
    logger.info(f"\n>>> Patient: {prompt}")
    if LF:
        # Start a span for this turn; attach input for observability
        with LF.start_as_current_span(name="ask", input={"user": USER_ID, "prompt": prompt}) as span:
            # Enrich the trace (user/session/tags). This sets properties on the **trace**, not only this span.
            span.update_trace(user_id=USER_ID, session_id=SESSION_ID, tags=["demo", "mem0-service"])
            # Run the agent
            result = await agent.run_async(prompt)
            span.update(output=result)
    else:
        result = await agent.run_async(prompt)

    logger.info(f"<<< Assistant: {result}")
    return result or "None"

# ------------------------------------------------------------------------------------
# Demo flow
# ------------------------------------------------------------------------------------
async def run_demo():
    if LF:
        # Create a root span to group the whole conversation
        with LF.start_as_current_span(
            name="demo_conversation",
            input={"app": APP_NAME, "model": MODEL, "mem0_base": MEM0_BASE_URL},
        ) as root:
            root.update_trace(user_id=USER_ID, session_id=SESSION_ID, tags=["demo", "healthcare"])
            await ask("Hi, I'm Alex. I've been having headaches for the past week, and I have a penicillin allergy.")
            await ask("Can you tell me more about what might be causing my headaches?")
            await ask("I think I should see a doctor. Can you help me schedule an appointment for next Monday at 2pm?")
            await ask("What medications should I avoid for my headaches?")
        try:
            LF.flush(); LF.shutdown()
        except Exception:
            pass
    else:
        await ask("Hi, I'm Alex. I've been having headaches for the past week, and I have a penicillin allergy.")
        await ask("Can you tell me more about what might be causing my headaches?")
        await ask("I think I should see a doctor. Can you help me schedule an appointment for next Monday at 2pm?")
        await ask("What medications should I avoid for my headaches?")

if __name__ == "__main__":
    asyncio.run(run_demo())
