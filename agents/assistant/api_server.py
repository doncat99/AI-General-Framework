# agents/assistant/api_server.py
from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel

from .agent import AssistantAgent
from .config import DEFAULT_USER_ID

app = FastAPI(title="ADK + mem0 Assistant", version="0.1.0")
agent = AssistantAgent()

class ChatRequest(BaseModel):
    user_id: str = DEFAULT_USER_ID
    message: str

class ChatResponse(BaseModel):
    response: str

@app.get("/")
async def root():
    return {"ok": True, "service": "adk-mem0-assistant"}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    out = await agent.ainvoke(req.user_id, req.message)
    return ChatResponse(response=out)
