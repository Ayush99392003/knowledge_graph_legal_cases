"""
Knowledge Graph Pipeline - Main orchestrator.

Runs the full document-to-KG workflow:
  1. PDF extraction & preprocessing
  2. LLM-based chunk processing for KG
  3. Post-processing & centralization
  4. Visualization

Usage:
    python pipeline.py [--pdf-path DATA.PDF]
"""

import logging
import sys
from pathlib import Path

from ko_pre_processing import process_pre
from llm_kg_creation import process_chunks
from kg_postprocessing import postprocess_llm_file
from build_kg_visual import build
from pp import post
# from createtp import build_relation_tuples
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main(pdf_path) -> int:
    """
    Execute the complete knowledge graph pipeline.

    Args:
        pdf_path: Path to the PDF file to process.

    Returns:
        0 on success, 1 on failure.
    """
    try:
        logger.info("Starting Knowledge Graph Pipeline")
        logger.info("Step 1: PDF extraction & preprocessing...")
        print(f"Processing PDF: {pdf_path}")
        process_pre(pdf_path)

        logger.info("Step 2: LLM-based KG extraction...")
        process_chunks()

        logger.info("Step 3: Post-processing & entity centralization...")
        postprocess_llm_file(
            "kg_output.json",
            output_kg_file="knowledge_graph.json",
            output_tuples_file="kg_tuples.json",
            output_origin_file="entity_origin_map.json",
        )

        logger.info("Step 4: Final post-processing...")
        post()

        logger.info("Step 5: Building visualization...")
        build()
        # build_relation_tuples()
        logger.info("✓ Pipeline completed successfully!")
        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    pdf_file = "data5.pdf"
    if len(sys.argv) > 1:
        pdf_file = sys.argv[1]
    sys.exit(main(pdf_file))
