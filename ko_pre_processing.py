"""Document preprocessing: PDF extraction, cleaning, and chunking."""

import json
import logging
import os
from typing import Dict, List

import pytesseract

from pdf_extraction import pdf_extraction
from processing import processing

logger = logging.getLogger(__name__)


def split_into_chunks(text: str, max_chars: int = 7000) -> List[Dict[str, str]]:
    """
    Split long text into LLM-friendly chunks at clean boundaries.

    Args:
        text: Text to chunk.
        max_chars: Max characters per chunk.

    Returns:
        List of chunk dicts with "text" key.
    """
    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = start + max_chars

        if end < n:
            # Prefer chunk break at newline or space
            newline_pos = text.rfind("\n", start, end)
            space_pos = text.rfind(" ", start, end)
            split_pos = max(newline_pos, space_pos)

            if split_pos <= start:
                split_pos = end
        else:
            split_pos = n

        block = text[start:split_pos].strip()
        if block:
            chunks.append({"text": block})

        start = split_pos

    return chunks


class DocumentProcessor:
    """Orchestrates PDF extraction, cleaning, and chunking."""

    def __init__(self, poppler_path: str = None, tesseract_path: str = None):
        """
        Initialize processor.

        Args:
            poppler_path: Path to poppler binary.
            tesseract_path: Path to tesseract executable.
        """
        self.POLARBRIEF_VERSION = "PolarBrief v1.0"
        self.poppler_path = poppler_path
        self.tesseract_path = tesseract_path
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

    def process_document(
        self, pdf_path: str, poppler_path: str = None, tesseract_path: str = None
    ) -> List[Dict[str, str]]:
        """
        Execute full preprocessing pipeline on PDF.

        Args:
            pdf_path: Path to PDF file.
            poppler_path: Optional override for poppler path.
            tesseract_path: Optional override for tesseract path.

        Returns:
            List of final LLM-ready chunks.

        Raises:
            FileNotFoundError: If PDF not found.
            Exception: On processing errors.
        """
        try:
            # Step 1: PDF → text extraction
            logger.info("Step 1: PDF extraction & OCR fallback...")
            extracted_lines = pdf_extraction(pdf_path, poppler_path, tesseract_path)

            merged_text = []

            # Step 2: Normalize OCR output
            for item in extracted_lines:
                if isinstance(item, str):
                    merged_text.append(item)
                elif isinstance(item, dict):
                    for key in ("text", "line", "content"):
                        if key in item and isinstance(item[key], str):
                            merged_text.append(item[key])
                            break

            # Build single merged text block
            chunks = [{"text": "\n".join(merged_text)}]
            logger.info("✓ Merged document into a single block")

            # Step 3: Full document cleaning
            logger.info("Step 2: Text processing & normalization...")
            processed_chunks = processing(chunks)
            logger.info("✓ Processing pipeline complete")

            full_clean_text = processed_chunks[0]["text"]

            # Step 4: Split into LLM-friendly chunks
            logger.info("Step 3: Splitting into LLM chunks...")
            final_chunks = split_into_chunks(full_clean_text, max_chars=25000)
            logger.info(f"✓ Split into {len(final_chunks)} LLM-friendly chunks")

            # Step 5: Save intermediate files
            logger.info("Step 4: Saving intermediate files...")
            self._save_chunks(chunks, processed_chunks, final_chunks)

            return final_chunks

        except FileNotFoundError as e:
            logger.error(f"PDF file not found: {pdf_path}")
            raise
        except Exception as e:
            logger.error(f"Processing failed: {e}", exc_info=True)
            raise

    @staticmethod
    def _save_chunks(
        raw_chunks: List[Dict],
        processed_chunks: List[Dict],
        final_chunks: List[Dict],
    ) -> None:
        """Save chunk files to disk."""
        try:
            with open("chunks_raw.json", "w", encoding="utf-8") as f:
                json.dump(raw_chunks, f, indent=4, ensure_ascii=False)

            with open("chunks_processed.json", "w", encoding="utf-8") as f:
                json.dump(processed_chunks, f, indent=4, ensure_ascii=False)

            with open("chunks_final.json", "w", encoding="utf-8") as f:
                json.dump(final_chunks, f, indent=4, ensure_ascii=False)

            logger.info(
                "✓ Saved: chunks_raw.json, chunks_processed.json, chunks_final.json"
            )
        except IOError as e:
            logger.error(f"Failed to save chunk files: {e}")
            raise


def process_pre(pdf_path: str) -> List[Dict[str, str]]:
    """
    Main entry point: preprocess a PDF document.

    Args:
        pdf_path: Path to PDF file.

    Returns:
        List of final chunks ready for LLM processing.
    """
    poppler_path = os.getenv("POPPLER_PATH")
    tesseract_path = os.getenv("TESSERACT_PATH")

    processor = DocumentProcessor(
        poppler_path=poppler_path, tesseract_path=tesseract_path
    )

    results = processor.process_document(
        pdf_path=pdf_path,
        poppler_path=poppler_path,
        tesseract_path=tesseract_path,
    )

    return results
process_pre("data5.pdf")