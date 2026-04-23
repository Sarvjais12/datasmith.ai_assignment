import logging
import anthropic

logger = logging.getLogger(__name__)

# Strict classification constraints for the LLM
SENTIMENT_DIRECTIVE = """
Analyze the sentiment. Return exactly: Label, Confidence %, Justification (1 sentence), and Tone.
--- Input Data ---
{input_payload}
"""

def analyze_sentiment(raw_text: str, client: anthropic.Anthropic) -> str:
    """
    Executes the sentiment analysis pipeline.
    Enforces strict template formatting and catches upstream API failures.
    """
    if not raw_text or not raw_text.strip():
        return "⚠️ Error: No text provided for sentiment analysis."

    # Cap length to maintain low latency and prevent token waste
    safe_payload = raw_text[:5000]
    
    compiled_prompt = SENTIMENT_DIRECTIVE.format(input_payload=safe_payload)

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=250,
            temperature=0.1,  # Low temp for stable, repeatable classification
            messages=[{"role": "user", "content": compiled_prompt}],
        )
        return response.content[0].text.strip()
        
    except anthropic.APIError as api_err:
        logger.error(f"Anthropic API rejected the sentiment classification: {api_err}")
        return f"⚠️ Sentiment analysis failed due to an upstream API error: {str(api_err)}"
    except Exception as e:
        logger.exception("Unexpected pipeline crash in analyze_sentiment.")
        return "⚠️ System error: Could not complete sentiment classification at this time."