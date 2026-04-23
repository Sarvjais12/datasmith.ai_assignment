import logging
import anthropic

logger = logging.getLogger(__name__)

# Strict RAG constraints to prevent hallucination and ensure high accuracy
RAG_DIRECTIVE = """
Answer the user's question using ONLY this document. If the answer isn't in there, just say 'The document doesn't contain this information.' Do not hallucinate.

--- Context Document ---
{context_payload}

--- User Query ---
{query_payload}
"""

def answer_question(doc_context: str, user_query: str, client: anthropic.Anthropic) -> str:
    """
    Executes a grounded Retrieval-Augmented Generation (RAG) query.
    Ensures the LLM relies solely on the provided document context to prevent hallucination.
    """
    if not doc_context or not doc_context.strip():
        return "⚠️ Error: No document content available to search against."
        
    if not user_query or not user_query.strip():
        return "⚠️ Error: No question was provided."

    # Truncate context to prevent context window overflow while maintaining latency
    safe_context = doc_context[:7500] 
    
    compiled_prompt = RAG_DIRECTIVE.format(
        context_payload=safe_context, 
        query_payload=user_query
    )

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=800,
            temperature=0.0,  # Absolute zero temperature forces strict adherence to the text
            messages=[{"role": "user", "content": compiled_prompt}],
        )
        return response.content[0].text.strip()
        
    except anthropic.APIError as api_err:
        logger.error(f"Anthropic API rejected the RAG query: {api_err}")
        return f"⚠️ Q&A search failed due to an upstream API error: {str(api_err)}"
    except Exception as e:
        logger.exception("Unexpected pipeline crash in answer_question.")
        return "⚠️ System error: Could not process the document query at this time."