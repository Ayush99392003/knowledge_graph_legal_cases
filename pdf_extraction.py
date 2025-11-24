"""PDF extraction with OCR fallback using pdfplumber and pytesseract."""

import logging
import re
from typing import Dict, List

import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

logger = logging.getLogger(__name__)


def pdf_extraction(
    pdf_path: str, poppler_path: str = None, tesseract_path: str = None
) -> List[Dict[str, str]]:
    """
    Extract text from PDF using pdfplumber, with OCR fallback for scanned pages.

    Args:
        pdf_path: Path to PDF file.
        poppler_path: Path to poppler binary (for pdf2image).
        tesseract_path: Path to tesseract executable.

    Returns:
        List of extracted text blocks with metadata.
    """
    print(f"Processing PDF: {pdf_path}")
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path

    def clean_text(text: str) -> str:
        """Clean OCR/extracted text."""
        text = re.sub(r"[•·●♦▪•∙]", "", text)
        text = re.sub(r"[^\w\s,.:;()\"'-]", "", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()

    def is_noisy(text: str, threshold: float = 0.6) -> bool:
        """Check if text is too noisy (many non-alphanumeric chars)."""
        if not text:
            return True
        non_alpha = sum(1 for c in text if not c.isalnum())
        return (non_alpha / len(text)) > threshold

    def has_repeated_characters(text: str, repeat_threshold: int = 4) -> bool:
        """Check for excessive repeated characters."""
        return bool(
            re.search(r"(.)\1{" + str(repeat_threshold) + ",}", text)
        )

    def extract_text_with_fallback(pdf_path: str) -> List[Dict[str, str]]:
        """Extract text with pdfplumber, OCR fallback for scanned pages."""
        final_output = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                for page_num, page in enumerate(pdf.pages, start=1):
                    logger.info(
                        f"Page {page_num}/{total_pages}: Trying PDF text extraction..."
                    )
                    page_line_no = 1
                    text = page.extract_text()
                    lines = text.split("\n") if text else []

                    if lines and sum(len(l.strip()) for l in lines) > 20:
                        for line in lines:
                            cleaned = line.strip()
                            if cleaned:
                                final_output.append(
                                    {
                                        "text": cleaned,
                                        "page_no": f"[p{page_num} {page_line_no}]",
                                        "method": "pdfplumber",
                                    }
                                )
                                page_line_no += 1
                        continue

                    logger.info(f"Page {page_num} has no usable text — fallback to OCR")
                    try:
                        images = convert_from_path(
                            pdf_path,
                            dpi=300,
                            first_page=page_num,
                            last_page=page_num,
                            poppler_path=poppler_path,
                        )
                        img = images[0]
                        gray = img.convert("L")
                        bw = gray.point(lambda x: 0 if x < 180 else 255, "1")
                        ocr_text = pytesseract.image_to_string(bw, lang="eng")
                        lines = ocr_text.strip().split("\n")
                        line_no = 1
                        for line in lines:
                            cleaned_line = clean_text(line)
                            if (
                                cleaned_line
                                and not is_noisy(cleaned_line)
                                and not has_repeated_characters(cleaned_line)
                            ):
                                final_output.append(
                                    {
                                        "text": cleaned_line,
                                        "page_no": f"[p{page_num} {line_no}]",
                                        "method": "ocr",
                                    }
                                )
                                line_no += 1
                    except Exception as e:
                        logger.warning(f"OCR failed for page {page_num}: {e}")

        except FileNotFoundError as e:
            logger.error(f"PDF file not found: {pdf_path}")
            raise
        except Exception as e:
            logger.error(f"Failed during PDF processing: {e}", exc_info=True)
            raise

        return final_output

    return extract_text_with_fallback(pdf_path)
