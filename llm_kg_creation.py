"""LLM-based knowledge graph extraction from document chunks."""

import copy
import json
import logging
import re
from typing import Any, Dict, List, Tuple

from config import DEPLOYMENT, client

logger = logging.getLogger(__name__)

ID_PATTERN_TEMPLATE = r"^chunk{chunk_idx}-\d{{3}}$"


def extract_kg_for_chunk(text_block: str, chunk_idx: int) -> Dict[str, List[Dict[str, Any]]]:
    """
    Call LLM to extract KG entities and relations from a text chunk.

    Args:
        text_block: Text to process.
        chunk_idx: Chunk index for ID formatting.

    Returns:
        Dict with "entities" and "relations" keys. Empty lists if LLM fails.
    """
    prompt = f"""ROLE:
You are an information extraction system that builds small, high-quality Knowledge Graphs from text. 
You identify the most important entities and the strongest relations, not only legal ones. 
Avoid trivial details, but do not restrict extraction only to strictly legal concepts.

INSTRUCTION:
Extract a minimal Knowledge Graph from the given text.

RULES:
- Output STRICT JSON only.
- Entities must contain exactly:
  - id, name, type
- Relations must contain exactly:
  - source, relation, target
- Entity IDs MUST follow:
  - chunk{chunk_idx}-XXX (XXX = 3-digit sequence starting at 001)
  - Example: "chunk{chunk_idx}-001"
- Relation source and target MUST reference existing entity IDs.
- Capture only important entities and strong, meaningful relations.
- Maintain a minimal set of entities and relations.
- No explanations. JSON ONLY.

CONTEXT:
You will receive a chunk of text. Extract the most important people, organizations, concepts, events, objects, or actions, and link them using their strongest relations. Only include information that contributes materially to understanding the text, but do not limit yourself to strictly legal details.

FORMAT:
{{
  "entities": [
    {{"id": "chunk{chunk_idx}-001", "name": "Entity name", "type": "Entity type"}}
  ],
  "relations": [
    {{"source": "chunk{chunk_idx}-001", "relation": "relationship", "target": "chunk{chunk_idx}-002"}}
  ]
}}

TEXT:
{text_block}
"""
    try:
        response = client.responses.create(
            model=DEPLOYMENT,
            input=prompt,
        )

        output = response.output_text.strip()
        parsed = json.loads(output)
        parsed.setdefault("entities", [])
        parsed.setdefault("relations", [])
        return parsed
    except json.JSONDecodeError:
        logger.warning(f"LLM returned invalid JSON for chunk {chunk_idx}")
        return {"entities": [], "relations": []}
    except Exception as e:
        logger.error(f"LLM extraction failed for chunk {chunk_idx}: {e}", exc_info=True)
        return {"entities": [], "relations": []}


def validate_chunk_kg(
    kg: Dict[str, List[Dict[str, Any]]], chunk_idx: int
) -> Tuple[bool, List[str]]:
    """
    Validate LLM KG output.

    Args:
        kg: Knowledge graph dict.
        chunk_idx: Chunk index for pattern matching.

    Returns:
        (is_valid, list_of_issues)
    """
    issues = []
    entities = kg.get("entities", [])
    relations = kg.get("relations", [])

    id_pattern = re.compile(ID_PATTERN_TEMPLATE.format(chunk_idx=chunk_idx))

    ids = []
    seqs = []
    for ent in entities:
        eid = ent.get("id")
        if not eid:
            issues.append("Entity missing id.")
            continue
        if not id_pattern.match(eid):
            issues.append(
                f"Entity id '{eid}' does not match required pattern 'chunk{chunk_idx}-###'."
            )
        ids.append(eid)
        m = re.search(r"(\d{3})$", eid)
        if m:
            seqs.append(int(m.group(1)))
        else:
            seqs.append(None)

    if len(ids) != len(set(ids)):
        issues.append("Duplicate entity ids detected.")

    if seqs:
        if None in seqs:
            issues.append("Some entity ids do not include a valid 3-digit sequence.")
        else:
            seqs_sorted = sorted(seqs)
            expected = list(range(seqs_sorted[0], seqs_sorted[0] + len(seqs_sorted)))
            if seqs_sorted != expected or seqs_sorted[0] != 1:
                issues.append(
                    f"Entity sequence numbers not consecutive from 001: {seqs_sorted}."
                )

    id_set = set(ids)
    for rel in relations:
        src = rel.get("source")
        tgt = rel.get("target")
        if not src or not tgt:
            issues.append(f"Relation missing source/target: {rel}")
        else:
            if src not in id_set:
                issues.append(f"Relation source '{src}' not found among entities.")
            if tgt not in id_set:
                issues.append(f"Relation target '{tgt}' not found among entities.")

    is_valid = len(issues) == 0
    return is_valid, issues


def rewrite_ids_for_chunk(
    kg: Dict[str, List[Dict[str, Any]]],
    chunk_idx: int,
    prefix: str = "chunk",
    pad: int = 3,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, str], List[Dict[str, Any]]]:
    """
    Deterministically rewrite entity IDs to chunk{chunk_idx}-{seq:03d}.

    Preserves original id in '_orig_id'. Updates relations accordingly.

    Args:
        kg: Original KG dict.
        chunk_idx: Chunk index for ID format.
        prefix: Prefix for new IDs.
        pad: Padding for sequence numbers.

    Returns:
        (rewritten_kg, local_id_map, unmapped_relations)
    """
    rewritten = copy.deepcopy(kg)
    entities = rewritten.get("entities", [])
    relations = rewritten.get("relations", [])

    local_id_map: Dict[str, str] = {}
    unmapped_relations: List[Dict[str, Any]] = []

    for seq, ent in enumerate(entities, start=1):
        orig_id = ent.get("id") or f"NOID_{seq}"
        ent["_orig_id"] = orig_id
        new_id = f"{prefix}{chunk_idx}-{str(seq).zfill(pad)}"
        ent["id"] = new_id
        local_id_map[orig_id] = new_id

    for rel in relations:
        src = rel.get("source")
        tgt = rel.get("target")

        mapped_src = local_id_map.get(src)
        mapped_tgt = local_id_map.get(tgt)

        if mapped_src:
            rel["source"] = mapped_src
        else:
            rel["_unmapped_source"] = src
        if mapped_tgt:
            rel["target"] = mapped_tgt
        else:
            rel["_unmapped_target"] = tgt

        if rel.get("_unmapped_source") or rel.get("_unmapped_target"):
            unmapped_relations.append(rel)

    return rewritten, local_id_map, unmapped_relations


def process_chunks(
    input_json: str = "chunks_final.json",
    output_json: str = "kg_output.json",
    id_map_json: str = "id_map.json",
    pad: int = 3,
) -> None:
    """
    Process all chunks through LLM KG extraction.

    Args:
        input_json: Input chunks file.
        output_json: Output KG file.
        id_map_json: Output ID map file.
        pad: Padding for sequence numbers.
    """
    try:
        with open(input_json, "r", encoding="utf-8") as f:
            chunks = json.load(f)
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_json}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {input_json}: {e}")
        raise

    final_output = []
    global_id_map: Dict[str, str] = {}
    all_unmapped_relations = []

    logger.info(f"Processing {len(chunks)} chunks...")

    for idx, item in enumerate(chunks, start=1):
        page = item.get("page")
        citation_start = item.get("citation_start")
        citation_end = item.get("citation_end")
        text_block = item.get("text") or item.get("clean_text")

        if not text_block:
            logger.warning(f"Skipping empty chunk {idx}")
            continue

        logger.info(f"Processing chunk {idx}...")

        kg = extract_kg_for_chunk(text_block, chunk_idx=idx)

        is_valid, issues = validate_chunk_kg(kg, chunk_idx=idx)
        if is_valid:
            logger.info(f"✓ Chunk {idx} validated.")
            for ent in kg.get("entities", []):
                ent["_orig_id"] = ent.get("id")
                ent["_chunk_index"] = idx

            for ent in kg.get("entities", []):
                composite_key = f"chunk{idx}::{ent.get('_orig_id')}"
                global_id_map[composite_key] = ent.get("id")

            chunk_result = {
                "page": page,
                "citation_start": citation_start,
                "citation_end": citation_end,
                "entities": kg.get("entities", []),
                "relations": kg.get("relations", []),
            }
            final_output.append(chunk_result)
        else:
            logger.warning(f"Validation failed for chunk {idx}. Issues: {issues}")
            logger.info("Falling back to deterministic rewrite...")

            rewritten, local_map, unmapped = rewrite_ids_for_chunk(
                kg, chunk_idx=idx, pad=pad
            )

            for ent in rewritten.get("entities", []):
                ent["_chunk_index"] = idx

            for orig, new in local_map.items():
                composite_key = f"chunk{idx}::{orig}"
                global_id_map[composite_key] = new

            if unmapped:
                for ur in unmapped:
                    all_unmapped_relations.append(
                        {"chunk_index": idx, "page": page, "relation": ur}
                    )

            chunk_result = {
                "page": page,
                "citation_start": citation_start,
                "citation_end": citation_end,
                "entities": rewritten.get("entities", []),
                "relations": rewritten.get("relations", []),
            }
            final_output.append(chunk_result)

    try:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=4, ensure_ascii=False)

        with open(id_map_json, "w", encoding="utf-8") as f:
            json.dump(global_id_map, f, indent=2, ensure_ascii=False)

        if all_unmapped_relations:
            with open("unmapped_relations.json", "w", encoding="utf-8") as f:
                json.dump(all_unmapped_relations, f, indent=2, ensure_ascii=False)
            logger.warning(f"Found {len(all_unmapped_relations)} unmapped relations.")

        logger.info(f"✓ Saved KG to {output_json}")
        logger.info(f"✓ Saved ID map to {id_map_json}")
    except IOError as e:
        logger.error(f"Failed to save output files: {e}")
        raise


if __name__ == "__main__":
    process_chunks()
