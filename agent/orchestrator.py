import asyncio
import json
import logging
import re
from typing import Optional

import anthropic
from fastapi import UploadFile

from agent.tasks.code_explain import explain_code
from agent.tasks.qa import answer_question
from agent.tasks.sentiment import analyze_sentiment
from agent.tasks.summarize import summarize_text
from utils.audio_transcribe import transcribe_audio
from utils.image_ocr import extract_image_text
from utils.pdf_parser import extract_pdf_text
from utils.youtube_fetch import fetch_youtube_transcript

logger = logging.getLogger(__name__)

# Grouped allowed formats
VALID_IMAGES = {"jpg", "jpeg", "png", "gif", "bmp", "webp"}
VALID_AUDIO = {"mp3", "wav", "m4a", "ogg", "flac"}

# Core system prompt for the routing brain
ROUTER_PROMPT = """
You are the routing brain of an agentic workflow. 
Analyze the user's input and the extracted document text (if provided).
Return ONLY raw JSON. No markdown formatting, no backticks.

{
  "intent": "<summarize | sentiment | code_explain | qa | extract | youtube | conversational | unclear>",
  "confidence": <float 0.0 to 1.0>,
  "reasoning": "<brief justification>",
  "needs_clarification": <boolean>,
  "clarification_question": "<string or null - keep it under 15 words>"
}

Note: If multiple intents apply or the request is too vague, set needs_clarification to true and ask what they want to do.
"""

class AgentOrchestrator:
    def __init__(self):
        # We maintain two clients: one for native async routes, one for background threaded sync tasks
        # TODO: move API key to a proper .env file, hardcoding environment is bad practice for prod
        self.async_client = anthropic.AsyncAnthropic()
        self.sync_client = anthropic.Anthropic()
        
        # Sonnet cost estimates per 1M tokens (Input/Output)
        self.input_cost_per_m = 3.00 
        self.output_cost_per_m = 15.00

    def _estimate_cost(self, text_chunk: str, is_input: bool = True) -> tuple[int, float]:
        """Rough heuristic for token counting and cost estimation to save API calls."""
        if not text_chunk:
            return 0, 0.0
            
        # Approximation: 1 token ≈ 4 characters
        estimated_tokens = len(text_chunk) // 4 
        rate = self.input_cost_per_m if is_input else self.output_cost_per_m
        estimated_cost = (estimated_tokens / 1_000_000) * rate
        
        return estimated_tokens, round(estimated_cost, 5)

    async def process(self, message: str, file: Optional[UploadFile], session: dict) -> dict:
        action_logs = []
        raw_doc_content = ""

        # Check if we are waiting on the user to clarify a previous request
        is_clarifying = session.get("awaiting_clarification", False)
        
        if is_clarifying and session.get("pending_content"):
            raw_doc_content = session["pending_content"]
            session["awaiting_clarification"] = False
            action_logs.append("User clarified intent. Resuming workflow.")
            context_payload = f"Document Content:\n{raw_doc_content}\n\nUser Follow-up: {message}"
            
        else:
            # Handle new file uploads using modern pattern matching
            if file and file.filename:
                file_ext = file.filename.rsplit(".", 1)[-1].lower()
                file_bytes = await file.read()
                action_logs.append(f"Ingesting {file.filename} ({len(file_bytes)} bytes)")

                match file_ext:
                    case ext if ext in VALID_IMAGES:
                        action_logs.append("Triggering Vision pipeline...")
                        raw_doc_content = await asyncio.to_thread(extract_image_text, file_bytes)
                    case "pdf":
                        action_logs.append("Extracting PDF layers...")
                        raw_doc_content = await asyncio.to_thread(extract_pdf_text, file_bytes)
                    case ext if ext in VALID_AUDIO:
                        action_logs.append("Running local Whisper transcription...")
                        raw_doc_content = await asyncio.to_thread(transcribe_audio, file_bytes, ext)
                    case _:
                        return {
                            "response": f"Unsupported format: .{file_ext}. Please upload a valid image, PDF, or audio file.",
                            "logs": action_logs,
                        }
                
                action_logs.append(f"Extraction complete: {len(raw_doc_content)} characters pulled.")

            # Check for YouTube links in the text
            yt_link = re.search(r"(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)[\w\-]+", message)
            if yt_link:
                target_url = yt_link.group(0)
                if not target_url.startswith("http"):
                    target_url = f"https://{target_url}"
                    
                action_logs.append(f"Scraping YouTube captions for: {target_url}")
                raw_doc_content = await asyncio.to_thread(fetch_youtube_transcript, target_url)

            # Build the payload for the LLM
            context_payload = f"Document Content:\n{raw_doc_content}\n\nUser Query: {message}" if raw_doc_content else f"User Query: {message}"

        # Run Cost Estimator
        tokens, cost = self._estimate_cost(context_payload)
        action_logs.append(f"Cost Estimate (Input): ~{tokens} tokens (${cost:.5f})")

        # Determine user intent
        action_logs.append("Querying routing engine...")
        parsed_goal = await self._determine_intent(context_payload)
        target_task = parsed_goal.get("intent", "conversational")
        
        # Stop and ask the user if the request is too vague
        if parsed_goal.get("needs_clarification"):
            session["pending_content"] = raw_doc_content or message
            session["awaiting_clarification"] = True
            
            fallback_q = parsed_goal.get("clarification_question", "Could you clarify what you'd like me to do with this?")
            action_logs.append("Halt: Intent is ambiguous. Requesting clarification.")
            
            return {
                "response": fallback_q,
                "extracted_text": raw_doc_content,
                "task_type": "clarification",
                "awaiting_clarification": True,
                "plan": ["ingest_data", "halt_for_clarification"],
                "logs": action_logs,
            }

        # Build workflow plan and run the specific tool
        workflow_steps = self._generate_plan(target_task, bool(raw_doc_content))
        final_output = await self._run_task(target_task, raw_doc_content, message, action_logs)

        # Estimate output cost
        out_tokens, out_cost = self._estimate_cost(final_output, is_input=False)
        action_logs.append(f"Cost Estimate (Output): ~{out_tokens} tokens (${out_cost:.5f})")

        # Save brief context to memory
        session["history"].append({"role": "user", "content": message[:300]})
        session["history"].append({"role": "assistant", "content": final_output[:300]})

        return {
            "response": final_output,
            "extracted_text": raw_doc_content,
            "task_type": target_task,
            "awaiting_clarification": False,
            "plan": workflow_steps,
            "logs": action_logs,
        }

    async def _determine_intent(self, payload: str) -> dict:
        """Calls Claude to strictly format the routing JSON."""
        try:
            response = await self.async_client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=300,
                system=ROUTER_PROMPT,
                messages=[{"role": "user", "content": payload[:5000]}],
            )
            clean_text = response.content[0].text.strip()
            # Nuke any markdown that Claude might accidentally inject
            clean_text = re.sub(r"^```[a-zA-Z]*\n?", "", clean_text)
            clean_text = re.sub(r"\n?```$", "", clean_text)
            
            return json.loads(clean_text)
        except Exception as e:
            logger.error(f"Routing engine failed to parse intent: {e}")
            return {"intent": "conversational", "confidence": 0.0, "needs_clarification": False}

    @staticmethod
    def _generate_plan(task: str, has_doc: bool) -> list[str]:
        """Constructs the step-by-step array for the UI."""
        base = ["ingest_data"] if has_doc else []
        
        routings = {
            "summarize": ["evaluate_intent", "execute_summarization_pipeline"],
            "sentiment": ["evaluate_intent", "execute_sentiment_analysis"],
            "code_explain": ["evaluate_intent", "parse_syntax", "explain_logic", "run_bug_check"],
            "qa": ["evaluate_intent", "query_document_context"],
            "extract": ["return_raw_data"],
            "youtube": ["scrape_video_captions", "evaluate_intent", "execute_summarization_pipeline"],
            "conversational": ["generate_direct_response"],
        }
        return base + routings.get(task, ["generate_direct_response"])

    async def _run_task(self, task: str, doc_text: str, user_msg: str, logs: list[str]) -> str:
        """Routes the sanitized data to the correct external task module."""
        try:
            active_content = doc_text or user_msg

            # We use asyncio.to_thread to push synchronous client calls to background workers
            if task == "summarize":
                logs.append("Executing summarization ruleset...")
                return await asyncio.to_thread(summarize_text, active_content, self.sync_client)

            if task == "sentiment":
                logs.append("Calculating sentiment score...")
                return await asyncio.to_thread(analyze_sentiment, active_content, self.sync_client)

            if task == "code_explain":
                logs.append("Analyzing code logic and searching for bugs...")
                return await asyncio.to_thread(explain_code, active_content, self.sync_client)

            if task == "qa":
                logs.append("Running targeted RAG query...")
                return await asyncio.to_thread(answer_question, doc_text, user_msg, self.sync_client)

            if task == "extract":
                return f"**Raw Data Extraction**\n\n{doc_text}" if doc_text else "No parseable data found."

            if task == "youtube":
                if doc_text and "Could not" not in doc_text:
                    logs.append("Captions retrieved. Generating summary...")
                    video_summary = await asyncio.to_thread(summarize_text, doc_text, self.sync_client)
                    snippet = doc_text[:500] + ("..." if len(doc_text) > 500 else "")
                    return f"**Video Transcript Snippet**\n\n{snippet}\n\n---\n\n{video_summary}"
                return f"⚠️ Failed to retrieve captions. They might be disabled.\nDetails: {doc_text}"

            # Fallback
            logs.append("Bypassing specialized tools. Running standard chat...")
            return await self._standard_chat(user_msg)

        except Exception as e:
            logger.exception("Pipeline failure.")
            return f"⚠️ System encountered an error during execution: {str(e)}"

    async def _standard_chat(self, msg: str) -> str:
        resp = await self.async_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=800,
            system="You are a direct, technical assistant. No fluff.",
            messages=[{"role": "user", "content": msg}],
        )
        return resp.content[0].text