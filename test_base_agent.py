# tests/test_base_agent_healthcare.py
import asyncio
import os
from typing import Dict, Any, List
from dotenv import load_dotenv
from loguru import logger

from utilities.base.base_agent import BaseAgent

# ---- Minimal loguru setup (stdout only, compact) ----------------------------
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

# ---- Env & clients ----------------------------------------------------------
load_dotenv()
MEM0_API_KEY = os.getenv("MEM0_API_KEY")
if not MEM0_API_KEY:
    raise RuntimeError("MEM0_API_KEY is required in your environment")

from mem0 import MemoryClient  # imported after env load
mem0_client = MemoryClient(api_key=MEM0_API_KEY)

APP_NAME = "healthcare_assistant_app"
USER_ID = os.getenv("DEMO_USER_ID", "Alex")
SESSION_ID = os.getenv("DEMO_SESSION_ID", "session_001")

# this global lets tools know which user to write/read for
GLOBAL_USER_ID = USER_ID
def _current_user_id() -> str:
    return globals().get("GLOBAL_USER_ID", USER_ID)

# ---- Tool functions (type-hinted, explicit returns) -------------------------
def save_patient_info(information: str) -> Dict[str, Any]:
    """
    Save important patient info (symptoms, conditions, allergies, preferences).
    Returns a status object indicating success.
    """
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

def retrieve_patient_info(query: str) -> Dict[str, Any]:
    """
    Retrieve relevant patient info from memory.
    Returns up to 5 memories above a 0.7 similarity threshold.
    """
    uid = _current_user_id()
    logger.info(f"[tool] retrieve_patient_info(uid={uid}) ← {query}")
    results = mem0_client.search(query=query, user_id=uid, limit=5, threshold=0.7)
    items: List[Dict[str, Any]] = results.get("results", []) if isinstance(results, dict) else []
    memories = [it.get("memory") for it in items if isinstance(it, dict) and it.get("memory")]
    logger.info(f"[tool] retrieve_patient_info → {len(memories)} hit(s)")
    return {"status": "success" if memories else "no_results", "memories": memories, "count": len(memories)}

def schedule_appointment(date: str, time: str, reason: str) -> Dict[str, Any]:
    """
    Create a simple appointment confirmation object. (Toy scheduler.)
    """
    appt_id = f"APT-{abs(hash(date + time)) % 10000:04d}"
    logger.info(f"[tool] schedule_appointment ← {date=} {time=} {reason=}")
    return {
        "status": "success",
        "appointment_id": appt_id,
        "confirmation": f"Appointment scheduled for {date} at {time} for {reason}",
        "message": "Please arrive 15 minutes early to complete paperwork.",
    }

# ---- Build BaseAgent --------------------------------------------------------
MODEL_NAME = os.getenv(
    "DEMO_MODEL_NAME",
    # pick a tool-capable model that works well via OpenRouter+LiteLLM in ADK
    # (you can override with DEMO_MODEL_NAME)
    "anthropic/claude-3.5-sonnet"
)

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
    use_cache=False,   # keep off for live debugging
)

# ---- Conversation helper ----------------------------------------------------
async def ask(text: str) -> str:
    globals()["GLOBAL_USER_ID"] = USER_ID  # ensure tools use this user
    logger.info(f"\n>>> Patient: {text}")
    reply = await agent.run_async(text)
    logger.info(f"<<< Assistant: {reply}\n")
    return reply or ""

# ---- Demo conversation (mirrors your working example) -----------------------
async def run_demo():
    # 1) Patient shares info (should trigger save_patient_info)
    await ask("Hi, I'm Alex. I've been having headaches for the past week, and I have a penicillin allergy.")

    # 2) Ask for causes (agent may call retrieve_patient_info first)
    await ask("Can you tell me more about what might be causing my headaches?")

    # 3) Schedule an appointment (should call schedule_appointment)
    await ask("I think I should see a doctor. Can you help me schedule an appointment for next Monday at 2pm?")

    # 4) Medication question (agent should remember allergy context)
    await ask("What medications should I avoid for my headaches?")

if __name__ == "__main__":
    asyncio.run(run_demo())
