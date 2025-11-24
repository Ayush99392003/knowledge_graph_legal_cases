"""Post-processing: entity canonicalization and deduplication."""

import json
import logging
from collections import defaultdict
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def postprocess_llm_file(
    input_file: str,
    output_kg_file: str = "knowledge_graph.json",
    output_tuples_file: str = "kg_tuples.json",
    output_origin_file: str = "entity_origin_map.json",
) -> None:
    """
    Centralize entities, deduplicate, and build canonical KG.

    Reads LLM chunk outputs, merges entities by name, and updates relations
    to use canonical IDs.

    Args:
        input_file: Path to LLM output (list of chunk results).
        output_kg_file: Output KG file path.
        output_tuples_file: Output tuples file path.
        output_origin_file: Output origin map file path.
    """
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            chunks_output: List[Dict[str, Any]] = json.load(f)
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_file}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {input_file}: {e}")
        raise

    logger.info(f"Processing {len(chunks_output)} chunks...")

    # 1) Build canonical entities map: name_lower -> canonical_id
    canonical_map: Dict[str, str] = {}
    global_entities: List[Dict[str, Any]] = []
    id_counter = 1

    origin_map: Dict[str, Dict[str, List[Any]]] = defaultdict(
        lambda: {"original_ids": [], "chunks": []}
    )

    for chunk_idx, chunk in enumerate(chunks_output):
        for ent in chunk.get("entities", []):
            name = ent.get("name", "")
            if not name:
                continue
            name_key = name.strip().lower()

            if name_key not in canonical_map:
                canonical_id = f"e{id_counter:04d}"
                canonical_map[name_key] = canonical_id
                id_counter += 1

                global_entities.append(
                    {
                        "id": canonical_id,
                        "name": ent["name"].strip(),
                        "type": ent.get("type"),
                    }
                )

            cid = canonical_map[name_key]
            orig_id = ent.get("_orig_id")
            entity_chunk_idx = ent.get("_chunk_index", chunk_idx)

            if orig_id is not None and orig_id not in origin_map[cid]["original_ids"]:
                origin_map[cid]["original_ids"].append(orig_id)
            if (
                entity_chunk_idx is not None
                and entity_chunk_idx not in origin_map[cid]["chunks"]
            ):
                origin_map[cid]["chunks"].append(entity_chunk_idx)

    logger.info(f"Created {len(global_entities)} canonical entities")

    # 2) Process relations: convert local ids -> canonical ids
    global_relations: List[Dict[str, Any]] = []
    seen_relations = set()
    relation_chunks_map: Dict[tuple, List[int]] = defaultdict(list)

    for chunk_idx_for_rel, chunk in enumerate(chunks_output):
        local_to_canonical: Dict[str, str] = {}
        for ent in chunk.get("entities", []):
            local_id = ent.get("id")
            name = ent.get("name", "")
            if local_id is None or not name:
                continue
            name_key = name.strip().lower()
            canonical_id = canonical_map.get(name_key)
            if canonical_id:
                local_to_canonical[local_id] = canonical_id

        for rel in chunk.get("relations", []):
            src_local = rel.get("source")
            tgt_local = rel.get("target")
            rel_type = rel.get("relation")

            if not (src_local and tgt_local and rel_type):
                continue

            src_cid = local_to_canonical.get(src_local)
            tgt_cid = local_to_canonical.get(tgt_local)

            if not src_cid or not tgt_cid:
                continue

            key = (src_cid, rel_type, tgt_cid)
            if key not in seen_relations:
                seen_relations.add(key)
                global_relations.append(
                    {
                        "source": src_cid,
                        "relation": rel_type,
                        "target": tgt_cid,
                        "chunks": [],
                    }
                )

            if (
                chunk_idx_for_rel is not None
                and chunk_idx_for_rel not in relation_chunks_map[key]
            ):
                relation_chunks_map[key].append(chunk_idx_for_rel)

    logger.info(f"Deuplicated to {len(global_relations)} relations")

    # 3) Populate chunks into global_relations and build tuples_list
    tuples_list: List[List[Any]] = []
    rel_dict_map = {
        (r["source"], r["relation"], r["target"]): r for r in global_relations
    }

    for key, chunks in relation_chunks_map.items():
        src_cid, rel_type, tgt_cid = key
        chunks_sorted = sorted(chunks)
        rel_entry = rel_dict_map.get(key)
        if rel_entry is not None:
            rel_entry["chunks"] = chunks_sorted

        tuples_list.append([src_cid, rel_type, tgt_cid, chunks_sorted])

    for r in global_relations:
        if "chunks" not in r or r["chunks"] is None:
            r["chunks"] = []

    # 4) Prepare final KG
    knowledge_graph = {"entities": global_entities, "relations": global_relations}

    # 5) Write files
    try:
        with open(output_kg_file, "w", encoding="utf-8") as f:
            json.dump(knowledge_graph, f, ensure_ascii=False, indent=2)

        with open(output_tuples_file, "w", encoding="utf-8") as f:
            json.dump(tuples_list, f, ensure_ascii=False, indent=2)

        cleaned_origin_map = {}
        for cid, info in origin_map.items():
            cleaned_origin_map[cid] = {
                "original_ids": info["original_ids"],
                "chunks": sorted(info["chunks"]),
            }

        with open(output_origin_file, "w", encoding="utf-8") as f:
            json.dump(cleaned_origin_map, f, ensure_ascii=False, indent=2)

        logger.info(f"✓ Saved {output_kg_file}, {output_tuples_file}, {output_origin_file}")
    except IOError as e:
        logger.error(f"Failed to save output files: {e}")
        raise


if __name__ == "__main__":
    postprocess_llm_file(
        "kg_output.json",
        output_kg_file="knowledge_graph.json",
        output_tuples_file="kg_tuples.json",
        output_origin_file="entity_origin_map.json",
    )
