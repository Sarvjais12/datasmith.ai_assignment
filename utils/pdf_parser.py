import io
import logging

logger = logging.getLogger(__name__)

# Feature toggles for local dependencies
HAS_PYMUPDF = False
HAS_TESSERACT = False

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    logger.critical("PyMuPDF (fitz) is missing. PDF ingestion will fail.")

try:
    import pytesseract
    from PIL import Image
    HAS_TESSERACT = True
except ImportError:
    logger.warning("pytesseract or Pillow is missing. OCR fallback disabled.")


def extract_pdf_text(raw_stream: bytes) -> str:
    """
    Processes PDF buffers. Attempts native text layer extraction first.
    If a page appears to be a scanned image (low character count), it falls back to Tesseract OCR.
    """
    if not HAS_PYMUPDF:
        return "System Error: PDF processing library (PyMuPDF) is not installed."

    parsed_content = []
    
    try:
        # Load the document from the raw byte stream
        pdf_doc = fitz.open(stream=raw_stream, filetype="pdf")
        total_pages = len(pdf_doc)
        logger.info(f"Successfully loaded PDF with {total_pages} pages.")
    except Exception as e:
        logger.exception("Corrupted or unreadable PDF stream.")
        return f"⚠️ Failed to parse PDF file. Details: {str(e)}"

    for page_num, page in enumerate(pdf_doc, start=1):
        # Attempt fast native extraction first
        text_layer = page.get_text("text").strip()

        # Heuristic: If there are fewer than 50 characters, it's likely a scanned image page.
        # I picked 50 chars after testing - scanned pages usually have 0-10 selectable chars. This heuristic seems to work well enough for now.
        if text_layer and len(text_layer) > 50:
            parsed_content.append(f"--- Page {page_num} (Native) ---\n{text_layer}")
            
        elif HAS_TESSERACT:
            logger.info(f"Page {page_num} lacks a text layer. Triggering OCR fallback.")
            try:
                # Render the page to a high-res image buffer (300 DPI is standard for OCR)
                pixmap = page.get_pixmap(dpi=300) 
                img_buffer = Image.open(io.BytesIO(pixmap.tobytes("png")))
                
                scanned_text = pytesseract.image_to_string(img_buffer).strip()
                
                if scanned_text:
                    parsed_content.append(f"--- Page {page_num} (OCR) ---\n{scanned_text}")
                else:
                    parsed_content.append(f"--- Page {page_num} (Blank/No Text) ---")
            except Exception as ocr_err:
                logger.warning(f"OCR engine failed on page {page_num}: {ocr_err}")
                parsed_content.append(f"--- Page {page_num} (OCR Failure) ---")
        else:
            parsed_content.append(f"--- Page {page_num} (Scanned Image - OCR Disabled) ---")

    pdf_doc.close()
    
    if not parsed_content:
        return "Warning: The PDF was processed but no readable text could be extracted."
        
    # Inject a clean metadata header for the LLM to read
    header = f"DOCUMENT EXTRACTION REPORT | Total Pages: {total_pages}\n" + "="*50 + "\n\n"
    
    return header + "\n\n".join(parsed_content)