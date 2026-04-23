"""
Test Suite — Agentic Multi-Modal Assistant
==========================================

Run with:
    pytest tests/ -v

Covers:
  - Intent detection for all task types
  - Summarization output format validation
  - Sentiment analysis label validation
  - Code explanation structure validation
  - Q&A grounding check
  - Clarification trigger
  - YouTube URL detection
  - File type routing
  - Health endpoint
  - Session management
"""

import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# ── Make sure project root is on PYTHONPATH ───────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_anthropic_response(text: str):
    """Build a minimal mock that mimics anthropic.types.Message."""
    block = MagicMock()
    block.text = text
    msg = MagicMock()
    msg.content = [block]
    return msg


def _intent_json(intent: str, clarify: bool = False, question: str | None = None) -> str:
    return json.dumps({
        "intent":                 intent,
        "confidence":             0.95,
        "reasoning":              "test",
        "needs_clarification":    clarify,
        "clarification_question": question,
        "constraints":            {},
    })


# ─────────────────────────────────────────────────────────────────────────────
# 1. Intent Detection Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestIntentDetection(unittest.TestCase):
    """Verify _detect_intent returns the expected intent for various inputs."""

    def setUp(self):
        with patch("anthropic.Anthropic"):
            from agent.orchestrator import AgentOrchestrator
            self.orch = AgentOrchestrator()

    def _mock_intent(self, intent: str):
        self.orch.client.messages.create.return_value = _make_anthropic_response(
            _intent_json(intent)
        )

    def test_summarize_intent(self):
        self._mock_intent("summarize")
        result = self.orch._detect_intent("Please summarize this article.")
        self.assertEqual(result["intent"], "summarize")

    def test_sentiment_intent(self):
        self._mock_intent("sentiment")
        result = self.orch._detect_intent("What is the sentiment of this review?")
        self.assertEqual(result["intent"], "sentiment")

    def test_code_explain_intent(self):
        self._mock_intent("code_explain")
        result = self.orch._detect_intent("def foo(): pass\n\nExplain this code.")
        self.assertEqual(result["intent"], "code_explain")

    def test_qa_intent(self):
        self._mock_intent("qa")
        result = self.orch._detect_intent("What are the action items from the meeting?")
        self.assertEqual(result["intent"], "qa")

    def test_extract_intent(self):
        self._mock_intent("extract")
        result = self.orch._detect_intent("Extract the text from this image.")
        self.assertEqual(result["intent"], "extract")

    def test_conversational_intent(self):
        self._mock_intent("conversational")
        result = self.orch._detect_intent("Hello, how are you?")
        self.assertEqual(result["intent"], "conversational")

    def test_clarification_needed(self):
        self.orch.client.messages.create.return_value = _make_anthropic_response(
            _intent_json("unclear", clarify=True, question="What would you like me to do?")
        )
        result = self.orch._detect_intent("Here is some content.")
        self.assertTrue(result["needs_clarification"])
        self.assertIsNotNone(result["clarification_question"])

    def test_malformed_json_falls_back_to_conversational(self):
        self.orch.client.messages.create.return_value = _make_anthropic_response(
            "not valid json !!!"
        )
        result = self.orch._detect_intent("Some input")
        self.assertEqual(result["intent"], "conversational")

    def test_intent_strips_markdown_fences(self):
        self.orch.client.messages.create.return_value = _make_anthropic_response(
            "```json\n" + _intent_json("summarize") + "\n```"
        )
        result = self.orch._detect_intent("Summarize this.")
        self.assertEqual(result["intent"], "summarize")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Plan Builder Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPlanBuilder(unittest.TestCase):
    def setUp(self):
        with patch("anthropic.Anthropic"):
            from agent.orchestrator import AgentOrchestrator
            self.orch = AgentOrchestrator()

    def test_plan_with_content(self):
        plan = self.orch._build_plan("summarize", has_content=True)
        self.assertIn("extract_content", plan)
        self.assertIn("run_summarization", plan)

    def test_plan_without_content(self):
        plan = self.orch._build_plan("conversational", has_content=False)
        self.assertNotIn("extract_content", plan)

    def test_code_plan_has_bug_detection(self):
        plan = self.orch._build_plan("code_explain", has_content=True)
        self.assertIn("flag_bugs", plan)

    def test_unknown_intent_defaults_to_conversational(self):
        plan = self.orch._build_plan("nonexistent_intent", has_content=False)
        self.assertIn("conversational_llm_response", plan)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Task: Summarization
# ─────────────────────────────────────────────────────────────────────────────

class TestSummarizeTask(unittest.TestCase):
    def test_output_contains_required_sections(self):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_anthropic_response(
            "**1-Line Summary**\nThis is a summary.\n\n"
            "**Key Points**\n• Point one\n• Point two\n• Point three\n\n"
            "**5-Sentence Summary**\nSentence one. Two. Three. Four. Five."
        )
        from agent.tasks.summarize import summarize_text
        result = summarize_text("Sample content.", mock_client)
        self.assertIn("1-Line Summary", result)
        self.assertIn("Key Points", result)
        self.assertIn("5-Sentence Summary", result)

    def test_long_text_is_truncated_before_api_call(self):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_anthropic_response("OK")
        from agent.tasks.summarize import summarize_text
        long_text = "x" * 20_000
        summarize_text(long_text, mock_client)
        call_args = mock_client.messages.create.call_args
        prompt = call_args[1]["messages"][0]["content"]
        self.assertLessEqual(len(prompt), 8000)  # SUMMARIZE_PROMPT + 7000 chars


# ─────────────────────────────────────────────────────────────────────────────
# 4. Task: Sentiment Analysis
# ─────────────────────────────────────────────────────────────────────────────

class TestSentimentTask(unittest.TestCase):
    VALID_LABELS = {"Positive", "Negative", "Neutral", "Mixed"}

    def _run(self, label: str):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_anthropic_response(
            f"Label: {label}\nConfidence: 87%\n"
            "Justification: The text expresses a clear opinion.\n"
            "Emotional Tone: Analytical"
        )
        from agent.tasks.sentiment import analyze_sentiment
        return analyze_sentiment("Some text.", mock_client)

    def test_positive_label(self):
        r = self._run("Positive")
        self.assertIn("Positive", r)

    def test_negative_label(self):
        r = self._run("Negative")
        self.assertIn("Negative", r)

    def test_output_contains_confidence(self):
        r = self._run("Neutral")
        self.assertIn("Confidence", r)

    def test_output_contains_justification(self):
        r = self._run("Mixed")
        self.assertIn("Justification", r)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Task: Code Explanation
# ─────────────────────────────────────────────────────────────────────────────

class TestCodeExplainTask(unittest.TestCase):
    SAMPLE_CODE = "def add(a, b):\n    return a + b"

    def _run(self):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_anthropic_response(
            "Language: Python\n\n"
            "What It Does:\nAdds two numbers.\n\n"
            "Step-by-Step Breakdown:\n1. Takes two arguments.\n\n"
            "Bugs / Issues:\nNone detected\n\n"
            "Time Complexity: O(1)\n\n"
            "Suggestions:\nAdd type hints."
        )
        from agent.tasks.code_explain import explain_code
        return explain_code(self.SAMPLE_CODE, mock_client)

    def test_language_detected(self):
        self.assertIn("Python", self._run())

    def test_time_complexity_present(self):
        self.assertIn("O(1)", self._run())

    def test_bugs_section_present(self):
        self.assertIn("Bugs", self._run())

    def test_suggestions_present(self):
        self.assertIn("Suggestions", self._run())


# ─────────────────────────────────────────────────────────────────────────────
# 6. Task: Q&A
# ─────────────────────────────────────────────────────────────────────────────

class TestQATask(unittest.TestCase):
    CONTENT  = "The meeting agenda covers Q3 targets and hiring plans."
    QUESTION = "What are the action items?"

    def test_qa_returns_answer(self):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_anthropic_response(
            "The action items are: review Q3 targets and finalise hiring plans."
        )
        from agent.tasks.qa import answer_question
        result = answer_question(self.CONTENT, self.QUESTION, mock_client)
        self.assertGreater(len(result), 10)

    def test_qa_content_passed_to_api(self):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_anthropic_response("answer")
        from agent.tasks.qa import answer_question
        answer_question(self.CONTENT, self.QUESTION, mock_client)
        prompt = mock_client.messages.create.call_args[1]["messages"][0]["content"]
        self.assertIn("Q3 targets", prompt)
        self.assertIn("action items", prompt)


# ─────────────────────────────────────────────────────────────────────────────
# 7. YouTube URL Detection
# ─────────────────────────────────────────────────────────────────────────────

class TestYouTubeURLDetection(unittest.TestCase):
    def test_standard_url(self):
        import re
        pattern = r"(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)[\w\-]+"
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        self.assertIsNotNone(re.search(pattern, url))

    def test_short_url(self):
        import re
        pattern = r"(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)[\w\-]+"
        url = "https://youtu.be/dQw4w9WgXcQ"
        self.assertIsNotNone(re.search(pattern, url))

    def test_non_youtube_url(self):
        import re
        pattern = r"(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)[\w\-]+"
        url = "https://vimeo.com/123456"
        self.assertIsNone(re.search(pattern, url))

    def test_video_id_extraction(self):
        from utils.youtube_fetch import _extract_video_id
        self.assertEqual(
            _extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ"),
            "dQw4w9WgXcQ",
        )
        self.assertEqual(
            _extract_video_id("https://youtu.be/dQw4w9WgXcQ"),
            "dQw4w9WgXcQ",
        )

    def test_invalid_url_returns_none(self):
        from utils.youtube_fetch import _extract_video_id
        self.assertIsNone(_extract_video_id("https://google.com"))


# ─────────────────────────────────────────────────────────────────────────────
# 8. Orchestrator — Clarification Flow (async)
# ─────────────────────────────────────────────────────────────────────────────

class TestClarificationFlow(unittest.IsolatedAsyncioTestCase):
    async def test_sets_awaiting_clarification(self):
        with patch("anthropic.Anthropic"):
            from agent.orchestrator import AgentOrchestrator
            orch = AgentOrchestrator()

        orch.client.messages.create.return_value = _make_anthropic_response(
            _intent_json("unclear", clarify=True, question="What should I do with this?")
        )

        session = {"history": [], "pending_content": None, "awaiting_clarification": False}
        result  = await orch.process(message="Here is some content.", file=None, session=session)

        self.assertTrue(result["awaiting_clarification"])
        self.assertTrue(session["awaiting_clarification"])
        self.assertIn("?", result["response"])

    async def test_resumes_after_clarification(self):
        with patch("anthropic.Anthropic"):
            from agent.orchestrator import AgentOrchestrator
            orch = AgentOrchestrator()

        # First call returns intent=summarize (after clarification resolved)
        orch.client.messages.create.return_value = _make_anthropic_response(
            _intent_json("summarize")
        )

        session = {
            "history": [],
            "pending_content": "Some document text.",
            "awaiting_clarification": True,
        }

        with patch("agent.orchestrator.summarize_text", return_value="SUMMARY"):
            result = await orch.process(message="Please summarize", file=None, session=session)

        self.assertFalse(result["awaiting_clarification"])
        self.assertEqual(result["task_type"], "summarize")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Orchestrator — File Routing (async)
# ─────────────────────────────────────────────────────────────────────────────

class TestFileRouting(unittest.IsolatedAsyncioTestCase):
    def _make_file(self, filename: str, content: bytes = b"data"):
        f = AsyncMock()
        f.filename = filename
        f.read     = AsyncMock(return_value=content)
        return f

    async def _run(self, filename: str, extractor_patch: str, intent: str):
        with patch("anthropic.Anthropic"):
            from agent.orchestrator import AgentOrchestrator
            orch = AgentOrchestrator()

        orch.client.messages.create.return_value = _make_anthropic_response(
            _intent_json(intent)
        )

        session = {"history": [], "pending_content": None, "awaiting_clarification": False}
        file    = self._make_file(filename)

        with patch(extractor_patch, return_value="Extracted content") as mock_fn, \
             patch("agent.orchestrator.summarize_text",  return_value="S"), \
             patch("agent.orchestrator.analyze_sentiment", return_value="S"), \
             patch("agent.orchestrator.explain_code",      return_value="S"), \
             patch("agent.orchestrator.answer_question",   return_value="S"):
            result = await orch.process(message="process this", file=file, session=session)
            mock_fn.assert_called_once()

        return result

    async def test_pdf_routes_to_pdf_parser(self):
        r = await self._run("report.pdf", "agent.orchestrator.extract_pdf_text", "summarize")
        self.assertNotIn("Unsupported", r["response"])

    async def test_image_routes_to_ocr(self):
        r = await self._run("photo.jpg", "agent.orchestrator.extract_image_text", "extract")
        self.assertNotIn("Unsupported", r["response"])

    async def test_audio_routes_to_transcribe(self):
        r = await self._run("lecture.mp3", "agent.orchestrator.transcribe_audio", "summarize")
        self.assertNotIn("Unsupported", r["response"])

    async def test_unsupported_file_returns_error(self):
        with patch("anthropic.Anthropic"):
            from agent.orchestrator import AgentOrchestrator
            orch = AgentOrchestrator()

        session = {"history": [], "pending_content": None, "awaiting_clarification": False}
        file    = self._make_file("archive.zip", b"PKdata")
        result  = await orch.process(message="use this", file=file, session=session)
        self.assertIn("Unsupported", result["response"])


# ─────────────────────────────────────────────────────────────────────────────
# 10. FastAPI Endpoints
# ─────────────────────────────────────────────────────────────────────────────

class TestFastAPIEndpoints(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from httpx import AsyncClient, ASGITransport
        with patch("anthropic.Anthropic"), \
             patch("agent.orchestrator.AgentOrchestrator") as MockOrch:
            self.mock_orch = MockOrch.return_value
            self.mock_orch.process = AsyncMock(return_value={
                "response":               "Hello!",
                "extracted_text":         "",
                "task_type":              "conversational",
                "awaiting_clarification": False,
                "plan":                   ["conversational_llm_response"],
                "logs":                   ["Detecting intent…"],
            })
            import importlib
            import main as m
            importlib.reload(m)
            m.orchestrator = self.mock_orch
            self.client = AsyncClient(
                transport=ASGITransport(app=m.app), base_url="http://test"
            )

    async def test_health_endpoint(self):
        r = await self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "healthy")

    async def test_chat_returns_session_id(self):
        r = await self.client.post("/chat", data={"message": "Hello"})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn("session_id", body)
        self.assertIn("response",   body)

    async def test_chat_preserves_session_id(self):
        sid = "test-session-123"
        self.mock_orch.process = AsyncMock(return_value={
            "response": "Hi", "extracted_text": "", "task_type": "conversational",
            "awaiting_clarification": False, "plan": [], "logs": [],
        })
        r = await self.client.post("/chat", data={"message": "Hi", "session_id": sid})
        self.assertEqual(r.json()["session_id"], sid)

    async def asyncTearDown(self):
        await self.client.aclose()


# ─────────────────────────────────────────────────────────────────────────────
# Sample test cases from assignment spec
# ─────────────────────────────────────────────────────────────────────────────

class TestAssignmentSampleCases(unittest.IsolatedAsyncioTestCase):
    """
    Mirrors the three sample test cases from the assignment rubric.
    Uses mocks to verify correct pipeline routing without real API calls.
    """

    def _make_orch(self):
        with patch("anthropic.Anthropic"):
            from agent.orchestrator import AgentOrchestrator
            return AgentOrchestrator()

    async def test_pdf_action_items(self):
        """PDF with meeting notes + 'What are the action items?' → QA task."""
        orch = self._make_orch()
        orch.client.messages.create.return_value = _make_anthropic_response(
            _intent_json("qa")
        )
        session = {"history": [], "pending_content": None, "awaiting_clarification": False}

        file = AsyncMock()
        file.filename = "meeting_notes.pdf"
        file.read = AsyncMock(return_value=b"%PDF-1.4 fake content")

        with patch("agent.orchestrator.extract_pdf_text",  return_value="Notes: Review budgets. Hire 3 engineers."), \
             patch("agent.orchestrator.answer_question",    return_value="Action items: review budgets, hire engineers."):
            result = await orch.process("What are the action items?", file, session)

        self.assertEqual(result["task_type"], "qa")
        self.assertIn("action items", result["response"].lower())

    async def test_image_code_explain(self):
        """Image with code + 'Explain' → code_explain task."""
        orch = self._make_orch()
        orch.client.messages.create.return_value = _make_anthropic_response(
            _intent_json("code_explain")
        )
        session = {"history": [], "pending_content": None, "awaiting_clarification": False}

        file = AsyncMock()
        file.filename = "screenshot.png"
        file.read = AsyncMock(return_value=b"\x89PNG fake")

        with patch("agent.orchestrator.extract_image_text", return_value="def foo(): pass"), \
             patch("agent.orchestrator.explain_code",        return_value="This is a Python function."):
            result = await orch.process("Explain", file, session)

        self.assertEqual(result["task_type"], "code_explain")

    async def test_audio_transcription_and_summary(self):
        """Audio file → transcribe → summarize."""
        orch = self._make_orch()
        orch.client.messages.create.return_value = _make_anthropic_response(
            _intent_json("summarize")
        )
        session = {"history": [], "pending_content": None, "awaiting_clarification": False}

        file = AsyncMock()
        file.filename = "lecture.mp3"
        file.read = AsyncMock(return_value=b"fake audio bytes")

        with patch("agent.orchestrator.transcribe_audio", return_value="This is a lecture transcript."), \
             patch("agent.orchestrator.summarize_text",    return_value="**1-Line Summary**\nA lecture."):
            result = await orch.process("", file, session)

        self.assertEqual(result["task_type"], "summarize")
        self.assertIn("lecture", result["extracted_text"].lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
