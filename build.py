"""
visualize_kg_safe_improved.py

Improved, safer, and more modular KG visualization utility.
- Supports CLI (input/output paths, HTML/PNG options)
- Graceful handling of optional dependencies (pyvis, matplotlib)
- Preserves node/edge metadata in tooltips
- Produces interactive HTML (pyvis or vis.js fallback) and/or static PNG (matplotlib)
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import logging
from typing import Dict, Any

import networkx as nx

# Optional imports
try:
    from pyvis.network import Network
    HAVE_PYVIS = True
except ImportError:
    HAVE_PYVIS = False

try:
    import matplotlib.pyplot as plt
    from matplotlib import cm
    HAVE_MATPLOTLIB = True
except ImportError:
    HAVE_MATPLOTLIB = False

# Defaults
DEFAULT_INPUT = "relation_tuples.json"
DEFAULT_OUT_DIR = "sd/"
DEFAULT_HTML = "kg_visualization.html"
DEFAULT_PNG = "kg_visualization.png"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def safe_load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        logger.error("Input file not found: %s", path)
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def build_graph(kg):
    G = nx.MultiDiGraph()
    
    # Detect if kg is a list of entities
    entities = kg if isinstance(kg, list) else kg.get("entities", [])
    for e in entities:
        node_id = str(e.get("id", ""))
        if not node_id:
            continue
        label = e.get("name") or e.get("label") or node_id
        ntype = e.get("type", "Unknown")
        G.add_node(node_id, label=label, type=ntype, provenance=e.get("provenance"), meta=e)
    
    # Relations
    relations = [] if isinstance(kg, list) else kg.get("relations", [])
    for r in relations:
        src = str(r.get("source"))
        tgt = str(r.get("target"))
        if not (G.has_node(src) and G.has_node(tgt)):
            continue
        rel_type = r.get("relation") or "RELATED_TO"
        G.add_edge(src, tgt, rel_type=rel_type, meta=r)
    
    return G



def compute_visual_params(G: nx.MultiDiGraph, min_size: int = 120, max_size: int = 1600) -> Dict[str, Any]:
    degree = dict(G.degree())
    max_deg = max(degree.values()) if degree else 1
    node_sizes = {n: max(min_size, int(min_size + (max_size - min_size) * (d / max_deg))) for n, d in degree.items()}
    for n in G.nodes():
        node_sizes.setdefault(n, min_size)

    types = nx.get_node_attributes(G, "type")
    unique_types = sorted(set(types.values()))
    type_to_color = {}

    if HAVE_MATPLOTLIB and unique_types:
        cmap = cm.get_cmap("tab20", len(unique_types))
        for i, t in enumerate(unique_types):
            rgba = cmap(i)
            type_to_color[t] = '#%02x%02x%02x' % tuple(int(255 * x) for x in rgba[:3])
    else:
        fallback_palette = ["#8dd3c7","#ffffb3","#bebada","#fb8072","#80b1d3","#fdb462","#b3de69","#fccde5","#d9d9d9","#bc80bd"]
        for i, t in enumerate(unique_types):
            type_to_color[t] = fallback_palette[i % len(fallback_palette)]

    return {"node_sizes": node_sizes, "type_to_color": type_to_color, "labels": {n: G.nodes[n]["label"] for n in G.nodes()}, "types": types}


def generate_pyvis_html(G: nx.MultiDiGraph, outpath: str, params: Dict[str, Any]) -> None:
    if not HAVE_PYVIS:
        logger.warning("PyVis not installed, skipping interactive HTML generation.")
        return

    net = Network(height="1000px", width="100%", bgcolor="#ffffff", font_color="#222222", directed=True)
    net.force_atlas_2based()

    for n in G.nodes():
        ntype = G.nodes[n].get("type", "Unknown")
        title = f"<b>{params['labels'].get(n, n)}</b><br>Type: {ntype}<br>ID: {n}"
        net.add_node(n, label=params['labels'].get(n, n), title=title, color=params['type_to_color'].get(ntype, "#cccccc"))

    for u, v, data in G.edges(data=True):
        net.add_edge(u, v, title=data.get("rel_type", ""), label=data.get("rel_type", ""))

    net.write_html(outpath)
    logger.info("Interactive HTML written to %s", outpath)


def generate_matplotlib_png(G: nx.MultiDiGraph, outpath: str, params: Dict[str, Any]) -> None:
    if not HAVE_MATPLOTLIB:
        logger.warning("Matplotlib not installed, skipping PNG generation.")
        return

    pos = nx.spring_layout(G, k=0.5, iterations=50)
    node_colors = [params['type_to_color'].get(params['types'].get(n, "Unknown"), "#cccccc") for n in G.nodes()]
    node_sizes = [params['node_sizes'].get(n, 120) for n in G.nodes()]

    plt.figure(figsize=(16, 12))
    nx.draw(G, pos, with_labels=True, labels=params['labels'], node_color=node_colors,
            node_size=node_sizes, edge_color="gray", arrows=True, font_size=10)
    plt.axis('off')
    plt.savefig(outpath, bbox_inches='tight')
    plt.close()
    logger.info("Static PNG written to %s", outpath)


def main():
    parser = argparse.ArgumentParser(description="Knowledge Graph Visualization Tool")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT, help="Path to KG JSON file")
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR, help="Output directory")
    parser.add_argument("--html", action="store_true", help="Generate interactive HTML")
    parser.add_argument("--png", action="store_true", help="Generate static PNG")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    kg_data = safe_load_json(args.input)
    G = build_graph(kg_data)
    params = compute_visual_params(G)

    if args.html:
        generate_pyvis_html(G, os.path.join(args.out_dir, DEFAULT_HTML), params)
    if args.png:
        generate_matplotlib_png(G, os.path.join(args.out_dir, DEFAULT_PNG), params)


if __name__ == "__main__":
    main()
