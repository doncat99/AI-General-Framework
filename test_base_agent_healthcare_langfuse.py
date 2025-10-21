# test_base_agent_healthcare_langfuse.py
import os
import asyncio
from typing import Dict, Any, List
from dotenv import load_dotenv
from loguru import logger

from utilities.base.base_agent import BaseAgent

# ------------------ logging ------------------
logger.remove()
logger.add(
    sink=lambda m: print(m, end=""),
    level=os.getenv("LOG_LEVEL", "INFO"),
    backtrace=False,
    diagnose=False,
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> "
           "| <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
           "- <level>{message}</level>\n",
)

# ------------------ env ----------------------
load_dotenv()
MEM0_API_KEY = os.getenv("MEM0_API_KEY")
if not MEM0_API_KEY:
    raise RuntimeError("MEM0_API_KEY is required")

# --- Langfuse (optional, no-op if not configured) ---
ENABLE_LANGFUSE = bool(
    os.getenv("LANGFUSE_PUBLIC_KEY")
    and os.getenv("LANGFUSE_SECRET_KEY")
)

try:
    if ENABLE_LANGFUSE:
        from langfuse import Langfuse
        from langfuse.decorators import observe as lf_observe
        langfuse = Langfuse()  # uses LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST
        logger.info("[lf] Langfuse enabled")
    else:
        raise ImportError("Langfuse disabled or missing credentials")
except Exception as _e:
    def lf_observe(*_args, **_kwargs):
        # no-op decorator
        def deco(f): return f
        return deco
    langfuse = None
    logger.info("[lf] Langfuse not active (no credentials or package missing)")

# ------------------ mem0 ---------------------
from mem0 import MemoryClient
mem0_client = MemoryClient(api_key=MEM0_API_KEY)

APP_NAME = "healthcare_assistant_app"
USER_ID = os.getenv("DEMO_USER_ID", "Alex")
SESSION_ID = os.getenv("DEMO_SESSION_ID", "session_001")

GLOBAL_USER_ID = USER_ID
def _current_user_id() -> str:
    return globals().get("GLOBAL_USER_ID", USER_ID)

# ------------------ tools --------------------
@lf_observe(name="save_patient_info", as_type="tool")
def save_patient_info(information: str) -> Dict[str, Any]:
    uid = _current_user_id()
    logger.info(f"[tool] save_patient_info(uid={uid}) ← {information[:120]}...")
    resp = mem0_client.add(
        [{"role": "user", "content": information}],
        user_id=uid,
        run_id="healthcare_session",
        metadata={"type": "patient_information"},
    )
    ok = bool(resp)
    logger.info(f"[tool] save_patient_info → ok={ok}")
    return {"status": "success" if ok else "error", "saved": ok}

@lf_observe(name="retrieve_patient_info", as_type="tool")
def retrieve_patient_info(query: str) -> Dict[str, Any]:
    uid = _current_user_id()
    logger.info(f"[tool] retrieve_patient_info(uid={uid}) ← {query}")
    results = mem0_client.search(query=query, user_id=uid, limit=5, threshold=0.7)
    items: List[Dict[str, Any]] = results.get("results", []) if isinstance(results, dict) else []
    memories = [it.get("memory") for it in items if isinstance(it, dict) and it.get("memory")]
    logger.info(f"[tool] retrieve_patient_info → {len(memories)} hit(s)")
    return {"status": "success" if memories else "no_results", "memories": memories, "count": len(memories)}

@lf_observe(name="schedule_appointment", as_type="tool")
def schedule_appointment(date: str, time: str, reason: str) -> Dict[str, Any]:
    appt_id = f"APT-{abs(hash(date + time)) % 10000:04d}"
    logger.info(f"[tool] schedule_appointment ← date={date!r} time={time!r} reason={reason!r}")
    return {
        "status": "success",
        "appointment_id": appt_id,
        "confirmation": f"Appointment scheduled for {date} at {time} for {reason}",
        "message": "Please arrive 15 minutes early to complete paperwork.",
    }

# ------------------ agent --------------------
MODEL_NAME = os.getenv("DEMO_MODEL_NAME", "anthropic/claude-3.5-sonnet")

agent = BaseAgent(
    agent_name="healthcare_assistant",
    model_name=MODEL_NAME,  # BaseAgent will normalize to openrouter/<...> if needed
    instruction="""
You are a helpful Healthcare Assistant with memory capabilities.

Primary responsibilities:
1) Remember patient info using 'save_patient_info' when they share symptoms, conditions, or preferences.
2) Retrieve past patient info using 'retrieve_patient_info' when relevant.
3) Help schedule appointments using 'schedule_appointment'.

Guidelines:
- Be empathetic and professional; you are not a doctor (no diagnosis/treatment).
- Save important info (symptoms, conditions, allergies, preferences).
- Before re-asking, check if relevant info exists.
- For serious symptoms, advise consulting a healthcare professional.
- Keep all patient info confidential.
""".strip(),
    tools=[save_patient_info, retrieve_patient_info, schedule_appointment],
    app_name=APP_NAME,
    user_id=USER_ID,
    session_id=SESSION_ID,
    use_cache=False,
)

# ------------------ helpers -----------------
@lf_observe(name="agent_turn", as_type="chain")
async def ask(text: str) -> str:
    globals()["GLOBAL_USER_ID"] = USER_ID  # ensure tools use this user
    logger.info(f"\n>>> Patient: {text}")
    reply = await agent.run_async(text)
    logger.info(f"<<< Assistant: {reply}\n")
    return reply or ""

@lf_observe(name="demo_conversation", as_type="chain")
async def run_demo():
    await ask("Hi, I'm Alex. I've been having headaches for the past week, and I have a penicillin allergy.")
    await ask("Can you tell me more about what might be causing my headaches?")
    await ask("I think I should see a doctor. Can you help me schedule an appointment for next Monday at 2pm?")
    await ask("What medications should I avoid for my headaches?")

if __name__ == "__main__":
    asyncio.run(run_demo())
