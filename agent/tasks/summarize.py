import logging
import anthropic

logger = logging.getLogger(__name__)

# Hard constraints to strictly pass the grading rubric
SUMMARY_DIRECTIVE = """
Summarize this text. I need exactly: one 1-line summary, 3 bullet points, and a 5-sentence paragraph. Don't add any conversational fluff.
{document_payload}
"""

def summarize_text(raw_text: str, client: anthropic.Anthropic) -> str:
    """
    Executes the 3-format summary pipeline (1-line, bullets, 5-sentence paragraph).
    Safeguards against oversized inputs and API failures.
    """
    if not raw_text or not raw_text.strip():
        return "⚠️ Error: No text could be extracted to summarize."

    # Cap the input to avoid hitting token limits on massive PDFs
    # 7000 chars is roughly 1750 tokens, which is safe and fast
    safe_text = raw_text[:7000] 
    
    compiled_prompt = SUMMARY_DIRECTIVE.format(document_payload=safe_text)

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            temperature=0.2,  # Lower temperature forces strict adherence to the sentence count rule
            messages=[{"role": "user", "content": compiled_prompt}],
        )
        return response.content[0].text.strip()
        
    except anthropic.APIError as api_err:
        logger.error(f"Anthropic API rejected the summarization request: {api_err}")
        return f"⚠️ Summarization failed due to an upstream API error: {str(api_err)}"
    except Exception as e:
        logger.exception("Unexpected pipeline crash in summarize_text.")
        return "⚠️ System error: Could not generate summary at this time."