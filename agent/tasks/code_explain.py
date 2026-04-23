import logging
import anthropic

logger = logging.getLogger(__name__)

# Hard constraints for the code review pipeline
REVIEW_DIRECTIVE = """
Review this code. Tell me the language, explain what it does, find any bugs, and state the time complexity.
--- Target Source Code ---
{source_payload}
"""

def explain_code(raw_code: str, client: anthropic.Anthropic) -> str:
    """
    Executes the static code analysis and bug detection pipeline.
    Includes failsafes for payload size and upstream API timeouts.
    """
    if not raw_code or not raw_code.strip():
        return "⚠️ Error: No code provided for analysis."

    # Truncate to roughly ~1500 tokens to ensure we don't blow up the context window
    # while keeping enough logic intact for a meaningful review.
    safe_code = raw_code[:6000]
    
    compiled_prompt = REVIEW_DIRECTIVE.format(source_payload=safe_code)

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1500,
            temperature=0.1,  # Ultra-low temp for deterministic, analytical code reviews
            messages=[{"role": "user", "content": compiled_prompt}],
        )
        return response.content[0].text.strip()
        
    except anthropic.APIError as api_err:
        logger.error(f"Anthropic API rejected the code review request: {api_err}")
        return f"⚠️ Analysis failed due to an upstream API error: {str(api_err)}"
    except Exception as e:
        logger.exception("Unexpected pipeline crash in explain_code.")
        return "⚠️ System error: Could not complete code analysis at this time."