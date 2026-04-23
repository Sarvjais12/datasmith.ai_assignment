# Agentic Multi-Modal Assistant

> **DataSmith AI — Gen AI Engineer Assignment**
> Built to handle Text, Image, PDF, and Audio inputs using an autonomous orchestration layer.

This project implements an agentic workflow that doesn't just "chat"—it evaluates user intent, handles multi-modal data extraction, and executes specific task pipelines (like structured summarization or code review) based on what you need.

---

## Architecture & Flow
The system is built on a **FastAPI** backend with a completely non-blocking asynchronous architecture. You can find the full `architecture.svg` in the root.

**How it works:**
1. **Ingestion:** Raw bytes from PDFs, Images, or Audio are parsed via specialized utilities (`pdf_parser.py`, `image_ocr.py`, etc.).
2. **Routing:** The `AgentOrchestrator` sends a structured JSON prompt to **Claude Sonnet 4.6** to determine the user's "Intent."
3. **Clarification Gate:** If the user's intent is vague (e.g., just uploading an image without instructions), the agent halts and asks a specific follow-up question. This state is stored in the session memory.
4. **Execution:** Once clear, the agent builds a plan and executes the specific task module.
5. **Transparency:** Every response returns a `plan` (steps taken) and `logs` (real-time activity), satisfying the explainability requirement.

---

## Core Features & Rubric Coverage
* **Multi-Modal Support:** Native support for images (Vision OCR), PDFs (Text layer + OCR fallback), and Audio (Local Whisper transcription).
* **Strict Summarization:** All summaries follow the mandated 1-line + 3-bullet + 5-sentence format.
* **Cost Estimator (Bonus):** Integrated a heuristic-based cost estimator in `orchestrator.py` that calculates input/output costs per 1M tokens ($3.00/$15.00) before and after every call.
* **Async Concurrency:** The orchestrator uses `AsyncAnthropic` while wrapping heavy sync utilities (like OCR) in `asyncio.to_thread` to keep the FastAPI event loop responsive.

---

## Quick Start

### 1. Prerequisites
- Python 3.11+
- Tesseract OCR (`brew install tesseract` or `apt install tesseract-ocr`)
- An Anthropic API Key

### 2. Install & Run
```bash
# Setup env
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install deps
pip install -r requirements.txt

# Start Server
export ANTHROPIC_API_KEY="sk-ant-..."
uvicorn main:app --reload --port 8000

```
---

### Open http://localhost:8000 to use the chat interface.

---

### Project Map
main.py: Route handlers and session lifecycle management.

agent/orchestrator.py: The routing brain. Handles intent and cost tracking.

agent/tasks/: Modularized handlers for Summarization, Sentiment, QA, and Code Explanation.

utils/: Data extraction layer (PDF, OCR, Audio, YouTube).
```bash
main.py                 → Route handlers and session lifecycle
agent/orchestrator.py   → Routing brain, intent detection, cost tracking  
agent/tasks/            → Summarize, Sentiment, QA, Code Explain modules
utils/                  → PDF, OCR, Audio, YouTube extractors
```
---

### Known Technical Debt
Session Persistence: Currently uses an in-memory dictionary. For a production-level deployment, this should be moved to Redis or a persistent DB to handle server restarts.

OCR Fallback: The PDF parser uses a character-count heuristic (<50 chars) to trigger OCR. While effective, this could be optimized for mixed-content documents using a more granular layout analysis.

Audio Weights: Using Whisper base for speed/local testing. Larger models (medium/large) offer better accuracy but require significantly more VRAM.

---

### Testing
Run the suite with: pytest tests/ -v
The suite covers ~42 test cases including intent routing, session state, and edge-case handling.
