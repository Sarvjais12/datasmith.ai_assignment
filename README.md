# Agentic Multi-Modal Assistant

> **DataSmith AI — Gen AI Engineer Assignment** > Accepts Text · Image · PDF · Audio → understands intent → executes tasks autonomously.

---

## Architecture
I've included an `architecture.svg` file in the root directory that maps out the FastAPI routing, intent detection logic, and task execution flow. 

## Quick Start

### Prerequisites
- Python 3.11+
- `ANTHROPIC_API_KEY` in your environment
- Tesseract OCR installed locally (`brew install tesseract` or `apt install tesseract-ocr`)

### Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

export ANTHROPIC_API_KEY="your-api-key-here"