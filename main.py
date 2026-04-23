"""
Agentic Multi-Modal Assistant — FastAPI Entry Point
Supports: Text | Image (OCR) | PDF | Audio (STT)
"""

import uuid
import logging
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from agent.orchestrator import AgentOrchestrator

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Agentic Multi-Modal Assistant",
    description="Accepts Text / Image / PDF / Audio, understands intent, executes tasks.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store  (use Redis / DB in production)
# FIXME: this will leak memory and reset on server restart. Need to hook up Redis if this goes to production.
_sessions: dict[str, dict] = {}

orchestrator = AgentOrchestrator()


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    raise HTTPException(status_code=404, detail="UI not found")


@app.post("/chat")
async def chat(
    message: str = Form(default=""),
    session_id: str = Form(default=""),
    file: Optional[UploadFile] = File(default=None),
):
    """
    Core chat endpoint.

    - Accepts optional file upload + text message.
    - Returns agent response, extracted text, task type, plan, and execution logs.
    """
    if not session_id:
        session_id = str(uuid.uuid4())

    session = _sessions.setdefault(
        session_id,
        {"history": [], "pending_content": None, "awaiting_clarification": False},
    )

    logger.info("Chat request | session=%s | file=%s | msg_len=%d",
                session_id, file.filename if file else "none", len(message))

    result = await orchestrator.process(message=message, file=file, session=session)
    _sessions[session_id] = session

    return JSONResponse({
        "session_id": session_id,
        "response": result["response"],
        "extracted_text": result.get("extracted_text", ""),
        "task_type": result.get("task_type", ""),
        "awaiting_clarification": result.get("awaiting_clarification", False),
        "plan": result.get("plan", []),
        "logs": result.get("logs", []),
    })


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    _sessions.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}


@app.get("/health")
async def health():
    return {"status": "healthy", "sessions_active": len(_sessions)}


# ── Entry ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
