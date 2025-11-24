"""
KG Post-processing Pipeline
Author: ChatGPT (GPT-5 Thinking mini)
Description: Full pipeline to normalize, deduplicate and clean a JSON knowledge-graph like the one
provided by the user. Reads `input.json`, writes `cleaned_output.json`, `id_map.json` and `quarantine.json`.

Install required packages:
    pip install rapidfuzz python-dateutil networkx tqdm ftfy

LLM-based deduplication requires OpenAI API (via config.py).

Usage:
    python kg_postprocessing.py --input input.json --output cleaned_output.json

This script performs:
 - string normalization (unicode, whitespace, punctuation rules)
 - type canonicalization
 - date & statute normalization
 - exact + fuzzy + LLM-based semantic deduplication
 - relation normalization & attribute extraction
 - provenance and confidence bookkeeping
 - pruning and quarantine
 - outputs mapping old_id -> canonical_id

Note: tune thresholds at top of file. LLM deduplication may incur API costs.
"""

from __future__ import annotations
import argparse
import json
import re
import unicodedata
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

try:
    from dateutil.parser import parse as dateparse
except Exception:
    raise SystemExit('Please install python-dateutil: pip install python-dateutil')

# LLM-based semantic deduplication
USE_LLM_DEDUPE = True
LLM_DEDUPE_THRESHOLD = 0.80  # similarity threshold for LLM-based merging

try:
    import sys
    sys.path.insert(0, '.')
    from config import client
    OPENAI_CLIENT = client
except Exception:
    OPENAI_CLIENT = None
    USE_LLM_DEDUPE = False

# Fuzzy matching
try:
    from rapidfuzz import fuzz
except Exception:
    raise SystemExit('Please install rapidfuzz: pip install rapidfuzz')

import logging
logger = logging.getLogger(__name__)

# small progress bar
try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **k: x

# Graph utilities
try:
    import networkx as nx
except Exception:
    nx = None

# fix text
try:
    import ftfy
except Exception:
    ftfy = None


# -----------------------------
# Tunable thresholds & config
# -----------------------------
FUZZY_THRESHOLD_HIGH = 88  # above -> auto-merge
FUZZY_THRESHOLD_LOW = 75  # between low and high -> candidate for review
EMBEDDING_SIM_THRESHOLD = 0.75  # cosine similarity
BLOCK_KEY_CHARS = 4  # for blocking
MIN_LABEL_LENGTH = 2
MIN_CONFIDENCE = 0.25

# canonical types mapping (extend as needed)
TYPE_CANON_MAP = {
    # Courts / forums
    'court': 'Court',
    'tribunal': 'Court',
    'nclt': 'Court',
    'nclat': 'Court',
    'supreme court': 'Court',
    'high court': 'Court',
    'district court': 'Court',
    'family court': 'Court',
    'consumer forum': 'Court',
    'national green tribunal': 'Court',
    'ngt': 'Court',
    'tax tribunal': 'Court',
    'income tax appellate tribunal': 'Court',
    'itat': 'Court',
    'labour court': 'Court',
    'commissioner': 'Court',
    'bench': 'Court',
    # Persons / legal actors
    'judge': 'Person',
    'justice': 'Person',
    'magistrate': 'Person',
    'lawyer': 'Person',
    'advocate': 'Person',
    'counsel': 'Person',
    'solicitor': 'Person',
    'agent': 'Person',
    'litigant': 'Person',
    'party': 'Person',
    'petitioner': 'Person',
    'respondent': 'Person',
    'appellant': 'Person',
    'defendant': 'Person',
    'plaintiff': 'Person',
    'victim': 'Person',
    'complainant': 'Person',
    # Organizations / entities
    'company': 'Organization',
    'corporation': 'Organization',
    'firm': 'Organization',
    'government': 'Organization',
    'govt': 'Organization',
    'ministry': 'Organization',
    'department': 'Organization',
    'public sector undertaking': 'Organization',
    'psu': 'Organization',
    'state': 'Organization',
    'centre': 'Organization',
    'union': 'Organization',
    'authority': 'Organization',
    'commission': 'Organization',
    'board': 'Organization',
    'college': 'Organization',
    'university': 'Organization',
    # Statutes / legal instruments
    'statute': 'Statute',
    'act': 'Statute',
    'law': 'Statute',
    'regulation': 'Statute',
    'rule': 'Statute',
    'bye-law': 'Statute',
    'notification': 'Statute',
    'gazette notification': 'Statute',
    'constitution': 'Statute',
    'preamble': 'Statute',
    # Provisions / subparts
    'section': 'StatuteProvision',
    's': 'StatuteProvision',
    'sec': 'StatuteProvision',
    'clause': 'StatuteProvision',
    'sub-clause': 'StatuteProvision',
    'subsection': 'StatuteProvision',
    'article': 'StatuteProvision',
    'para': 'StatuteProvision',
    'paragraph': 'StatuteProvision',
    'schedule': 'StatuteProvision',
    'provision': 'StatuteProvision',
    # Case & decision types
    'case': 'Case',
    'appeal': 'Case',
    'revision': 'Case',
    'writ': 'Case',
    'criminal': 'Case',
    'civil': 'Case',
    'suo motu': 'Case',
    'petition': 'Case',
    'ba/baill': 'Case',  # placeholder mapping
    'review': 'Case',
    'inquiry': 'Case',
    'inquest': 'Case',
    # Document types & orders
    'courtorder': 'CourtOrder',
    'court order': 'CourtOrder',
    'interim order': 'CourtOrder',
    'final order': 'CourtOrder',
    'judgment': 'CourtOrder',
    'judgement': 'CourtOrder',
    'order': 'CourtOrder',
    'judicial order': 'CourtOrder',
    'memorandum': 'Document',
    'affidavit': 'Document',
    'plea': 'Document',
    'petition': 'Document',
    'notice': 'Document',
    'notice of motion': 'Document',
    'papers': 'Document',
    'transcript': 'Document',
    'casefile': 'Document',
    'court record': 'Document',
    # Roles & functions
    'legalrole': 'Role',
    'role': 'Role',
    'amicus curiae': 'Role',
    'trustee': 'Role',
    'guardian': 'Role',
    'receiver': 'Role',
    'petitioners counsel': 'Role',
    # Committees, panels
    'committee': 'Committee',
    'panel': 'Committee',
    'bench panel': 'Committee',
    # Misc / catch-all
    'reference': 'Reference',
    'citation': 'Citation',
    'evidence': 'Evidence',
    'exhibit': 'Exhibit',
    'schedule': 'Schedule',
}

# Extended canonical relation mapping (common legal relation variants)
RELATION_CANON_MAP = {
    "petitionerin": "petitioner",
    "petitioner": "petitioner",
    "respondentin": "respondent",
    "respondent": "respondent",
    "decidedby": "decided_by",
    "decided_by": "decided_by",
    "held": "held",
    "declared": "held",
    "delivered": "held",
    "amended": "amended",
    "amended_by": "amended",
    "amends": "amended",
    "inserted_into": "amended",
    "added_by": "amended",
    "cited": "cited",
    "citedin": "cited",
    "is_cited_in": "cited",
    "followed": "followed",
    "addresses": "issue",
    "addressed_issue": "issue",
    "considered_issue": "issue",
    "issue_in": "issue",
    # party relationships
    'is_appellant_in': 'is_appellant_in',
    'is_respondent_in': 'is_respondent_in',
    'is_petitioner_in': 'is_petitioner_in',
    'is_defendant_in': 'is_defendant_in',
    'is_plaintiff_in': 'is_plaintiff_in',
    'is_corporate_debtor_in': 'is_corporate_debtor_in',
    'is_creditor_in': 'is_creditor_in',
    # procedural actions
    'filed': 'filed',
    'filed_by': 'filed_by',
    'served': 'served',
    'served_on': 'served_on',
    'submitted': 'filed',
    'presented': 'filed',
    'appealed': 'appealed',
    'appealable_to': 'appealable_to',
    'dismissed': 'dismissed',
    'allowed': 'allowed',
    'passed': 'passed',
    'issued': 'issued',
    'granted': 'granted',
    'denied': 'denied',
    'struck_off': 'struck_off',
    'adjourned': 'adjourned',
    # representation / counsel
    'represented_by': 'represented_by',
    'counsel_for': 'represented_by',
    'appeared_for': 'represented_by',
    # citations & references
    'cites': 'cites',
    'is_cited_by': 'is_cited_by',
    'refers_to': 'refers_to',
    'relies_on': 'relies_on',
    'applies': 'applies',
    'interprets': 'interprets',
    'distinguishes': 'distinguishes',
    'overruled_by': 'overruled_by',
    'overrules': 'overrules',
    'upheld_by': 'upheld_by',
    'reversed_by': 'reversed_by',
    # statutory relationships
    'section_of': 'section_of',
    'clause_of': 'clause_of',
    'amended_by': 'amended_by',
    'enacted_by': 'enacted_by',
    'implements': 'implements',
    # remedies & outcomes
    'relief_granted': 'relief_granted',
    'award': 'award',
    'penalty_imposed': 'penalty_imposed',
    'sentenced': 'sentenced',
    'compensation_awarded': 'compensation_awarded',
    # administrative / regulatory
    'regulates': 'regulates',
    'prohibits': 'prohibits',
    'authorizes': 'authorizes',
    # fuzzy mapping examples (map words to canonical)
    'is_related_to': 'refers_to',
    'pertains_to': 'refers_to',
    'concerns': 'refers_to',
}

# patterns to remove from labels (more comprehensive)
REMOVE_PATTERNS = [
    # trailing role indicators like "(Respondent 1)", "(Appellant 2)" etc.
    r"\(respondent[s]?\s*\d*\)$",
    r"\(appellant[s]?\s*\d*\)$",
    r"\(petitioner[s]?\s*\d*\)$",
    r"\(defendant[s]?\s*\d*\)$",
    # "order dated 12-jan-2020", "order dated:", "order dated "
    r"^order dated\s*[:\-]?\s*",
    r"order dated\s*[:\-]?\s*\w+[\w\-\s\,]*$",  # if used as suffix
    # common prefixes/suffixes
    r"^judgment\s*[:\-]?\s*", 
    r"^judgement\s*[:\-]?\s*",
    r"^order\s*[:\-]?\s*",
    r"^final order\s*[:\-]?\s*",
    r"^interim order\s*[:\-]?\s*",
    # citations and reporter marks (simple forms)
    r"\bAIR\s*\d{4}\b.*",         # e.g., AIR 1995 SC 123
    r"\bSCC\b.*",                 # e.g., SCC 2003 456
    r"\b(200\d|201\d|202\d|199\d)\b.*",  # year-based trailing citation
    r"\b\d{1,4}\s+CLR\b.*",       # other reporters
    # "versus" or "v." noisy forms inside labels (keep canonical separate)
    r"\s+versus\s+.*$",
    r"\s+\bv\.?\b\s+.*$",
    # counsel/learned counsel mentions
    r",?\s*learned counsel for (the )?petitioners?$",
    r",?\s*learned counsel for (the )?respondents?$",
    r",?\s*through (its )?counsel$",
    # "respondent no. 1", "appellant no.2"
    r"\brespondent\s+no\.?\s*\d+\b",
    r"\bappellant\s+no\.?\s*\d+\b",
    # trailing dates
    r",?\s*dated\s*\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$",
    r",?\s*dated\s*[A-Za-z]{3,}\s*\d{1,2},?\s*\d{4}$",
    # parenthetical noise like "(through counsel)", "(in person)"
    r"\(through counsel\)$",
    r"\(in person\)$",
    r"\(for short\)$",
    # extra whitespace and separators
    r"^\s+|\s+$",
    r"\s+[-–—]\s+.*$",   # remove trailing dash and following noise
    # common prefixes to normalize away
    r"^the\s+",          # leading "the"
    # bracketed case numbers or registry marks: "[CIVIL APPEAL NO. 1234]"
    r"^\[.*\]$",
    # jurisdiction tags like "Supreme Court of India -"
    r"^(supreme court of india|high court of .+?)\s*[-:]\s*",
]


# -----------------------------
# Data classes
# -----------------------------

@dataclass
class Entity:
    id: str
    name: str
    type: Optional[str] = None
    label_norm: str = ''
    metadata: dict = field(default_factory=dict)
    provenance: list = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class Relation:
    source: str
    relation: str
    target: str
    chunks: list
    attributes: dict = field(default_factory=dict)
    provenance: list = field(default_factory=list)
    confidence: float = 1.0


# -----------------------------
# Helpers: normalization
# -----------------------------

def normalize_unicode(text: str) -> str:
    if text is None:
        return ''
    if ftfy:
        text = ftfy.fix_text(text)
    text = unicodedata.normalize('NFKC', text)
    return text


def clean_label(text: str) -> str:
    s = normalize_unicode(text)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    # remove known noisy patterns
    for p in REMOVE_PATTERNS:
        s = re.sub(p, '', s, flags=re.I).strip()
    # normalize dashes
    s = s.replace('\u2013', '-').replace('\u2014', '-')
    # remove repeated punctuation
    s = re.sub(r"[\s]*\(\s*\)$", '', s)
    return s


def canonicalize_type(typ: Optional[str]) -> str:
    if not typ:
        return 'Unknown'
    t = str(typ).strip().lower()
    t = re.sub(r'[^a-z0-9 ]', '', t)
    return TYPE_CANON_MAP.get(t, typ if typ in TYPE_CANON_MAP.values() else t.title())


# -----------------------------
# Statute & date parsing
# -----------------------------

def parse_date(text: str) -> Optional[str]:
    if not text:
        return None
    # try to find explicit date patterns first
    try:
        # remove words like 'Order dated '
        s = re.sub(r'order dated', '', text, flags=re.I)
        s = re.sub(r'[^0-9A-Za-z ,\-()]', ' ', s)
        dt = dateparse(s, fuzzy=True)
        return dt.strftime('%Y-%m-%d')
    except Exception:
        return None


def parse_statute(text: str) -> dict:
    out = {'raw': text}
    if not text:
        return out
    # simple extraction: statute name and section tokens
    m = re.search(r'(Section|Sec\.?)[\s]*(\d+[A-Za-z0-9\(\)\-]*)', text, flags=re.I)
    if m:
        out['section'] = m.group(2)
    m2 = re.search(r'of\s+([A-Za-z &.,0-9-]+)$', text, flags=re.I)
    if m2:
        out['statute'] = m2.group(1).strip()
    # fallback: find common statute acronyms
    if 'ibc' in text.lower():
        out['statute'] = 'IBC'
    if 'limitation' in text.lower():
        out['statute'] = out.get('statute', 'Limitation Act')
    return out


# -----------------------------
# Deduplication
# -----------------------------

def block_key(label: str, typ: str) -> str:
    k = label.lower()[:BLOCK_KEY_CHARS]
    return f"{k}__{typ}"


def fuzzy_score(a: str, b: str) -> float:
    """Combined fuzzy score using token_set_ratio and token_sort_ratio for better handling of extra tokens and order differences."""
    a = (a or '').lower()
    b = (b or '').lower()
    if not a or not b:
        return 0.0
    try:
        s1 = fuzz.token_set_ratio(a, b)
        s2 = fuzz.token_sort_ratio(a, b)
        score = 0.6 * s1 + 0.4 * s2
        return float(score)
    except Exception:
        return 0.0


def _extract_json_from_text(text: str):
    """Try to extract a JSON object/array from an LLM response string."""
    text = text.strip()
    # Try direct load first
    try:
        return json.loads(text)
    except Exception:
        pass
    # Attempt to find first JSON array or object
    start = None
    end = None
    # prefer array
    a = text.find('[')
    b = text.rfind(']')
    if a != -1 and b != -1 and b > a:
        start, end = a, b + 1
    else:
        a = text.find('{')
        b = text.rfind('}')
        if a != -1 and b != -1 and b > a:
            start, end = a, b + 1
    if start is not None and end is not None:
        candidate = text[start:end]
        try:
            return json.loads(candidate)
        except Exception:
            logger.debug('Could not parse extracted JSON candidate from LLM response')
            return None
    return None


def get_llm_batch_similarity(texts: List[str]) -> Dict[Tuple[int, int], float]:
    """Query LLM once to get pairwise similarity confidences for all unique pairs.

    Returns a dict mapping (i,j) -> confidence (0.0-1.0). If LLM fails, returns empty dict.
    """
    if not USE_LLM_DEDUPE or OPENAI_CLIENT is None or not texts:
        return {}

    M = len(texts)
    # build enumerated list for prompt
    enumerated = '\n'.join([f'{idx}: {t}' for idx, t in enumerate(texts)])
    prompt = (
        "You are given a numbered list of entity labels.\n"
        "For every unique pair of indices (i < j) return whether the two labels refer to the same entity.\n"
        "Output ONLY a JSON array of objects with fields: {\"i\": int, \"j\": int, \"similar\": true/false, \"confidence\": 0.0-1.0}.\n"
        "Confidence should be a number between 0 and 1 (1 = identical). Be concise and strict.\n\n"
        f"Labels:\n{enumerated}\n\n"
        "Return results for all pairs (i<j)."
    )

    try:
        # allow larger token budget for bigger pair counts
        max_tokens = 1000 if M <= 10 else min(3000, 200 + M * 150)
        response = OPENAI_CLIENT.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
         
        )

        response_text = response.choices[0].message.content.strip()
        parsed = _extract_json_from_text(response_text)
        if not parsed:
            logger.warning('LLM batch returned unparsable JSON for similarity matrix')
            return {}

        result: Dict[Tuple[int, int], float] = {}
        # Expect parsed to be a list of objects
        if isinstance(parsed, dict):
            # maybe the model returned an object with a 'pairs' key
            parsed = parsed.get('pairs', []) if 'pairs' in parsed else []

        if not isinstance(parsed, list):
            logger.warning('LLM batch returned unexpected JSON schema')
            return {}

        for item in parsed:
            try:
                i = int(item.get('i'))
                j = int(item.get('j'))
                conf = float(item.get('confidence', 0.0))
                if i >= j:
                    # normalize ordering
                    a, b = min(i, j), max(i, j)
                    i, j = a, b
                result[(i, j)] = max(0.0, min(1.0, conf))
            except Exception:
                continue
        return result
    except Exception as e:
        logger.warning(f'LLM batch similarity check failed: {e}')
        return {}


def dedupe_entities(entities: Dict[str, Entity]) -> Tuple[Dict[str, Entity],
                                                         Dict[str, str],
                                                         List[Tuple[str, str, float]]]:
    """
    Connected-component dedupe using fuzzy edges and optional embeddings.
    Returns (final_entities, id_map, review_candidates).
    Uses a UnionFind on entity IDs to ensure transitive merges.
    """
    ents = list(entities.values())
    if not ents:
        return {}, {}, []

    labels = [e.label_norm or '' for e in ents]
    types = [e.type or 'Unknown' for e in ents]

    # blocking to limit comparisons (use block_key helper)
    blocks = defaultdict(list)
    for idx, e in enumerate(ents):
        bk = block_key(e.label_norm.lower(), e.type or 'Unknown')
        blocks[bk].append(idx)

    # UnionFind over entity ids (strings)
    class UnionFind:
        def __init__(self):
            self.parent = {}
        def find(self, x):
            self.parent.setdefault(x, x)
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]
        def union(self, a, b):
            ra, rb = self.find(a), self.find(b)
            if ra == rb:
                return
            # deterministic: choose lexicographically smaller id as root
            if ra < rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb

    uf = UnionFind()
    review_candidates: List[Tuple[str, str, float]] = []

    # Stage A: within-block fuzzy comparisons -> union or candidate for review
    for bk, idxs in tqdm(blocks.items(), desc='fuzzy-blocks'):
        m = len(idxs)
        for i in range(m):
            for j in range(i + 1, m):
                ii, jj = idxs[i], idxs[j]
                a_label = (labels[ii] or '').strip().lower()
                b_label = (labels[jj] or '').strip().lower()
                if not a_label or not b_label:
                    continue
                s = fuzzy_score(a_label, b_label)
                id_a = ents[ii].id
                id_b = ents[jj].id
                if s >= FUZZY_THRESHOLD_HIGH:
                    uf.union(id_a, id_b)
                elif s >= FUZZY_THRESHOLD_LOW:
                    review_candidates.append((id_a, id_b, s))

    # Stage B: optional LLM-based merges across component representatives
    if USE_LLM_DEDUPE and OPENAI_CLIENT is not None:
        # build components as current UF groups
        comp_map = defaultdict(list)
        for e in ents:
            comp_map[uf.find(e.id)].append(e)
        rep_ids = list(comp_map.keys())
        rep_texts = [comp_map[r][0].label_norm for r in rep_ids]
        M = len(rep_ids)
        
        # Single LLM batch call to check all representative pairs at once
        logger.info(f"LLM-based deduplication: checking {M} component representatives (batch)")
        pair_conf = get_llm_batch_similarity(rep_texts)
        if not pair_conf:
            logger.info('LLM batch returned no results; skipping Stage B merges')
        else:
            for (i, j), conf in pair_conf.items():
                try:
                    if conf >= LLM_DEDUPE_THRESHOLD:
                        logger.info(f"LLM merge: {rep_texts[i]} <-> {rep_texts[j]} (similarity: {conf:.2f})")
                        uf.union(rep_ids[i], rep_ids[j])
                except Exception:
                    # ignore malformed index pairs
                    continue

    # finalize groups -> mapping root_id -> member ids
    comp_groups = defaultdict(list)
    for e in ents:
        root = uf.find(e.id)
        comp_groups[root].append(e.id)

    # choose canonical id and merge members' data
    id_map: Dict[str, str] = {}
    canonical: Dict[str, Entity] = {}
    for root, members in comp_groups.items():
        # choose canonical id deterministically (lexicographically smallest)
        canonical_id = sorted(members)[0]
        base = deepcopy(entities[canonical_id])
        # merge others into base
        for mid in members:
            if mid == canonical_id:
                continue
            other = entities[mid]
            # provenance: extend, avoid exact duplicates
            for p in other.provenance:
                if p not in base.provenance:
                    base.provenance.append(p)
            # confidence: take the max (more confident source wins)
            base.confidence = max(base.confidence, other.confidence)
            # metadata: keep existing keys; add new ones from other
            for k, v in other.metadata.items():
                if k not in base.metadata:
                    base.metadata[k] = deepcopy(v)
            id_map[mid] = canonical_id
        id_map[canonical_id] = canonical_id
        canonical[canonical_id] = base

    # ensure mapping for any originals that somehow missed mapping
    for orig_id in entities.keys():
        id_map.setdefault(orig_id, orig_id)

    # build final_entities dictionary
    final_entities = {cid: deepcopy(ent) for cid, ent in canonical.items()}

    return final_entities, id_map, review_candidates


# -----------------------------
# Relation normalization & updates
# -----------------------------
# -----------------------------
# Relation deduplication
# -----------------------------
def dedupe_relations(relations: List[Relation],
                     ignore_chunk_order: bool = True,
                     attrs_ignore_keys: Optional[List[str]] = None,
                     fuzzy_high: int = FUZZY_THRESHOLD_HIGH,
                     fuzzy_low: int = FUZZY_THRESHOLD_LOW,
                     use_llm: bool = True) -> List[Relation]:
    """
    Robust relation deduplication:
      - exact canonicalization-based collapse
      - fuzzy merging within blocks (rapidfuzz)
      - optional LLM-guided merging across ambiguous components

    Returns deduplicated Relation objects (deepcopies), merging provenance/confidence/attributes/chunks.
    """
    if attrs_ignore_keys is None:
        attrs_ignore_keys = ['provenance', 'created_at', 'updated_at', 'timestamp', 'source']

    # helpers ---------------------------------------------------------
    def _normalize_str(s: Optional[str]) -> str:
        if s is None:
            return ''
        s2 = normalize_unicode(str(s)).strip()
        s2 = re.sub(r'\s+', ' ', s2)
        return s2

    def _attrs_for_key(attrs: dict) -> dict:
        """Return a stable, minimal attributes dict used for equality / hashing."""
        out = {}
        for k, v in (attrs or {}).items():
            if k in attrs_ignore_keys:
                continue
            # normalize scalars/lists/dicts into deterministic primitives
            if isinstance(v, list):
                out[k] = tuple([str(x) for x in v])
            elif isinstance(v, dict):
                try:
                    out[k] = tuple(sorted((str(kk), str(vv)) for kk, vv in v.items()))
                except Exception:
                    out[k] = str(v)
            else:
                out[k] = str(v)
        return out

    def _attrs_key_json(attrs: dict) -> str:
        try:
            return json.dumps(_attrs_for_key(attrs), sort_keys=True, ensure_ascii=False)
        except Exception:
            return str(sorted((_ for _ in _attrs_for_key(attrs).items())))

    def _chunks_key(chunks):
        if not chunks:
            return tuple()
        if ignore_chunk_order:
            try:
                return tuple(sorted([str(c) for c in chunks]))
            except Exception:
                return tuple(str(c) for c in chunks)
        else:
            return tuple(str(c) for c in chunks)

    # Nothing to do
    if not relations:
        return []

    # Build canonical metadata for each relation, and an index -> relation mapping
    rel_items = []
    for idx, r in enumerate(relations):
        src = _normalize_str(getattr(r, 'source', None))
        tgt = _normalize_str(getattr(r, 'target', None))
        rel_label, extra_attrs = normalize_relation_label(str(getattr(r, 'relation', '') or ''))
        attrs = deepcopy(r.attributes or {})
        # include any attrs produced by normalization into the canonical key
        if extra_attrs:
            # do not mutate original attributes; use a copy
            attrs.update(extra_attrs)
        label_for_key = rel_label
        attrs_json = _attrs_key_json(attrs)
        chunks_key = _chunks_key(getattr(r, 'chunks', []))
        # store normalized text used for fuzzy/LLM later
        text_repr = f"relation: {label_for_key}; source: {src}; target: {tgt}; attrs: {attrs_json}"
        rel_items.append({
            'idx': idx,
            'src': src,
            'tgt': tgt,
            'rel_label': label_for_key,
            'attrs': attrs,
            'attrs_json': attrs_json,
            'chunks_key': chunks_key,
            'text_repr': text_repr,
            'orig': r
        })

    # Stage 1: exact-key collapse into buckets
    buckets = {}
    for it in rel_items:
        key = (it['src'], it['rel_label'], it['tgt'], it['attrs_json'], it['chunks_key'])
        buckets.setdefault(key, []).append(it['idx'])

    # Map each relation index to a group id (initially each bucket -> a group)
    # We'll use UnionFind to allow fuzzy/LLM merges to union groups.
    class UF:
        def __init__(self):
            self.parent = {}
        def find(self, x):
            self.parent.setdefault(x, x)
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]
        def union(self, a, b):
            ra, rb = self.find(a), self.find(b)
            if ra == rb:
                return
            # choose deterministically smaller root
            if ra < rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb

    uf = UF()
    # initialize groups: pick representative index (smallest idx) as group root for bucket
    bucket_reps = []
    for key, idxs in buckets.items():
        rep = min(idxs)
        bucket_reps.append(rep)
        for i in idxs:
            uf.union(rep, i)  # union all members to representative

    # Stage 2: fuzzy merging within source-target blocks to avoid n^2
    # Build blocks by (src[:k], tgt[:k], rel_label) to restrict comparisons
    BLOCK_K = 8
    blocks = defaultdict(list)
    for it in rel_items:
        block_key = (it['src'][:BLOCK_K].lower(), it['tgt'][:BLOCK_K].lower(), it['rel_label'])
        blocks[block_key].append(it)

    # fuzzy compare only representatives of current groups to avoid redundant checks
    # We'll compare per-block representatives
    for blk_key, items in tqdm(blocks.items(), desc='relation-fuzzy-blocks'):
        m = len(items)
        for i in range(m):
            for j in range(i + 1, m):
                a = items[i]
                b = items[j]
                # skip if already in same union group
                if uf.find(a['idx']) == uf.find(b['idx']):
                    continue
                # compute fuzzy on the combined textual representation (rel label + source + target)
                s = fuzzy_score(a['text_repr'], b['text_repr'])
                if s >= fuzzy_high:
                    uf.union(a['idx'], b['idx'])
                elif s >= fuzzy_low:
                    # borderline case -> if LLM available will be handled later, otherwise leave as separate
                    # mark candidate by leaving them separate; we'll optionally pass to LLM stage
                    pass

    # Stage 3: LLM-based merging across current group representatives (optional)
    if use_llm and USE_LLM_DEDUPE and OPENAI_CLIENT is not None:
        # Build current components (map root -> member indices)
        comp_map = defaultdict(list)
        for it in rel_items:
            comp_map[uf.find(it['idx'])].append(it)

        rep_roots = list(comp_map.keys())
        rep_texts = []
        rep_indices = []
        for root in rep_roots:
            # choose canonical representative tuple to summarize component:
            members = comp_map[root]
            # pick the one with the longest text_repr (heuristic) as representative
            rep = max(members, key=lambda x: len(x['text_repr']))
            # create a concise natural-language description for LLM judgement:
            # include relation label, source and target, and top attrs (stringified)
            txt = f"Relation: {rep['rel_label']}\nSource: {rep['src']}\nTarget: {rep['tgt']}\nAttrs: {_attrs_for_key(rep['attrs'])}"
            rep_texts.append(txt)
            rep_indices.append(root)

        # only call LLM if there's at least 2 reps
        if len(rep_texts) >= 2:
            logger.info(f"LLM relation dedupe: checking {len(rep_texts)} representatives")
            pair_conf = get_llm_batch_similarity(rep_texts)
            if pair_conf:
                for (i, j), conf in pair_conf.items():
                    try:
                        if conf >= LLM_DEDUPE_THRESHOLD:
                            # union the underlying indices (root ids are relation indices)
                            uf.union(rep_indices[i], rep_indices[j])
                    except Exception:
                        continue
            else:
                logger.info("LLM returned no usable relation similarity results; skipping LLM merges")
    else:
        if use_llm and (not USE_LLM_DEDUPE or OPENAI_CLIENT is None):
            logger.info("LLM dedupe disabled or OpenAI client not available; skipping LLM stage for relations")

    # Stage 4: Build final groups and merge relation objects
    groups = defaultdict(list)
    for it in rel_items:
        root = uf.find(it['idx'])
        groups[root].append(it['idx'])

    # For each group produce a canonical merged Relation
    merged_relations: List[Relation] = []
    for root_idx, member_idxs in groups.items():
        # choose canonical index deterministically: smallest original idx
        canonical_idx = min(member_idxs)
        base_rel = deepcopy(relations[canonical_idx])
        # ensure canonicalized label/attrs/chunks are consistent
        base_rel.relation, norm_extra = normalize_relation_label(base_rel.relation or '')
        # ensure attributes dict exists
        base_rel.attributes = deepcopy(base_rel.attributes or {})
        base_rel.attributes.update(norm_extra or {})

        # collect unique provenance in preserved order
        prov_seen = []
        for midx in sorted(member_idxs):
            rsrc = relations[midx]
            for p in getattr(rsrc, 'provenance', []):
                if p not in prov_seen:
                    prov_seen.append(deepcopy(p))
        base_rel.provenance = prov_seen

        # confidence: max of members
        max_conf = 0.0
        for midx in member_idxs:
            try:
                max_conf = max(max_conf, float(getattr(relations[midx], 'confidence', 0.0) or 0.0))
            except Exception:
                continue
        base_rel.confidence = max_conf or getattr(base_rel, 'confidence', 0.0)

        # merge attributes conservatively
        for midx in member_idxs:
            other = relations[midx]
            for k, v in (other.attributes or {}).items():
                if k not in base_rel.attributes:
                    base_rel.attributes[k] = deepcopy(v)
                else:
                    ev = base_rel.attributes[k]
                    # both lists -> append unique
                    if isinstance(ev, list) and isinstance(v, list):
                        base_rel.attributes[k] = ev + [x for x in v if x not in ev]
                    elif isinstance(ev, list) and not isinstance(v, list):
                        if v not in ev:
                            base_rel.attributes[k] = ev + [v]
                    elif not isinstance(ev, list) and isinstance(v, list):
                        combined = [ev] + [x for x in v if x != ev]
                        base_rel.attributes[k] = combined
                    else:
                        if ev != v:
                            # preserve both as a list to avoid data loss
                            base_rel.attributes[k] = [ev, v] if ev != v else ev

        # merge chunks: ordered unique
        merged_chunks = []
        for midx in sorted(member_idxs):
            ch = getattr(relations[midx], 'chunks', []) or []
            for c in ch:
                if c not in merged_chunks:
                    merged_chunks.append(c)
        base_rel.chunks = merged_chunks

        # normalize endpoints to strings (keep original id strings)
        base_rel.source = _normalize_str(base_rel.source)
        base_rel.target = _normalize_str(base_rel.target)

        merged_relations.append(base_rel)

    # final pass: sanity de-duplication exact-by-(source,relation,target,attrs) to be safe
    final_seen = {}
    final_list = []
    for r in merged_relations:
        key = (str(_normalize_str(r.source)), str(r.relation), str(_normalize_str(r.target)), _attrs_key_json(r.attributes))
        if key in final_seen:
            # merge with existing
            existing = final_seen[key]
            # provenance
            for p in (r.provenance or []):
                if p not in existing.provenance:
                    existing.provenance.append(deepcopy(p))
            existing.confidence = max(existing.confidence, r.confidence)
            # attributes merging (same logic)
            for k, v in (r.attributes or {}).items():
                if k not in existing.attributes:
                    existing.attributes[k] = deepcopy(v)
                else:
                    ev = existing.attributes[k]
                    if isinstance(ev, list) and isinstance(v, list):
                        existing.attributes[k] = ev + [x for x in v if x not in ev]
                    elif isinstance(ev, list) and not isinstance(v, list):
                        if v not in ev:
                            existing.attributes[k] = ev + [v]
                    elif not isinstance(ev, list) and isinstance(v, list):
                        combined = [ev] + [x for x in v if x != ev]
                        existing.attributes[k] = combined
                    else:
                        if ev != v:
                            existing.attributes[k] = [ev, v]
            # chunks
            for c in r.chunks or []:
                if c not in existing.chunks:
                    existing.chunks.append(c)
        else:
            final_seen[key] = deepcopy(r)
            final_list.append(final_seen[key])

    return final_list



def normalize_relation_label(rel: str) -> Tuple[str, dict]:
    # split noisy compound words
    r = rel.strip()
    r = r.replace('-', '_')
    r = re.sub(r'[^A-Za-z0-9_]', '_', r)
    r = re.sub(r'__+', '_', r).strip('_').lower()
    attr = {}
    # collapse long verbs
    if 'condonation' in r and 'dismiss' in r:
        r = 'dismissed'
        attr['subject'] = 'condonation_application'
    # map to canonical if present
    r_can = RELATION_CANON_MAP.get(r, r)
    return r_can, attr


def update_relations(relations: List[Relation], id_map: Dict[str, str], entities: Dict[str, Entity]) -> List[Relation]:
    new_rels = []
    for r in relations:
        src = id_map.get(r.source, r.source)
        tgt = id_map.get(r.target, r.target)
        rel_label, attrs = normalize_relation_label(r.relation)
        # attach provenance and maintain chunks
        new_r = deepcopy(r)
        new_r.source = src
        new_r.target = tgt
        new_r.relation = rel_label
        new_r.attributes.update(attrs)
        # confidence: lower if either endpoint id changed (indicates uncertainty)
        if r.source != src or r.target != tgt:
            new_r.confidence = max(0.01, new_r.confidence * 0.9)
        # check endpoints exist in final entities
        if src not in entities or tgt not in entities:
            new_r.attributes['invalid_endpoint'] = True
        new_rels.append(new_r)
    return new_rels


# -----------------------------
# Build objects from input json
# -----------------------------

def load_input(path: str) -> Tuple[Dict[str, Entity], List[Relation]]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    ent_map = {}
    for e in data.get('entities', []):
        ent = Entity(id=e.get('id'), name=e.get('name'), type=e.get('type'))
        ent.label_norm = clean_label(ent.name)
        ent.metadata['original'] = e
        # provenance: capture full original entity
        ent.provenance.append({'source': 'input', 'raw': e})
        ent.type = canonicalize_type(ent.type)
        ent.confidence = 0.9
        ent_map[ent.id] = ent
    rels = []
    for r in data.get('relations', []):
        rel = Relation(source=r.get('source'), relation=r.get('relation'), target=r.get('target'), chunks=r.get('chunks', []))
        rel.provenance.append({'source': 'input', 'raw': r})
        rel.confidence = 0.9
        rels.append(rel)
    return ent_map, rels


# -----------------------------
# Add provenance + confidence
# -----------------------------

def enrich_entities(entities: Dict[str, Entity]):
    for e in entities.values():
        # ensure canonical label and short label
        e.metadata['label_canonical'] = e.label_norm
        e.metadata['label_original'] = e.name
        # if missing type info
        if not e.type or e.type == 'Unknown':
            # attempt to infer from name heuristics
            if re.search(r'\b(tribunal|court|bench|nclt|nclat|supreme)\b', e.name, flags=re.I):
                e.type = 'Court'
            elif re.search(r'\b(limited|ltd|company|corporation|inc\b)\b', e.name, flags=re.I):
                e.type = 'Organization'
            else:
                e.type = 'Unknown'
        # ensure provenance exists
        if 'created_at' not in e.metadata:
            e.metadata['created_at'] = datetime.utcnow().isoformat() + 'Z'


# -----------------------------
# Pruning / quarantine
# -----------------------------

def prune_relations(relations: List[Relation], entities: Dict[str, Entity]) -> Tuple[List[Relation], List[Relation]]:
    kept = []
    quarantined = []
    for r in relations:
        if r.source == r.target and r.attributes.get('invalid_endpoint'):
            quarantined.append(r)
            continue
        # drop if endpoints missing
        if r.source not in entities or r.target not in entities:
            r.confidence *= 0.5
            quarantined.append(r)
            continue
        # drop very low confidence
        if r.confidence < MIN_CONFIDENCE:
            quarantined.append(r)
            continue
        kept.append(r)
    return kept, quarantined


# -----------------------------
# Output writers
# -----------------------------

def to_serializable_entities(entities: Dict[str, Entity]) -> List[dict]:
    out = []
    for e in entities.values():
        o = {
            'id': e.id,
            'name': e.name,
            'type': e.type,
            'label_canonical': e.metadata.get('label_canonical'),
            'label_original': e.metadata.get('label_original'),
            'provenance': e.provenance,
            'confidence': e.confidence,
            'metadata': {k: v for k, v in e.metadata.items() if k not in ('label_canonical','label_original')}
        }
        out.append(o)
    return out


def to_serializable_relations(relations: List[Relation]) -> List[dict]:
    out = []
    for r in relations:
        out.append({
            'source': r.source,
            'relation': r.relation,
            'target': r.target,
            'attributes': r.attributes,
            'provenance': r.provenance,
            'confidence': r.confidence,
            'chunks': r.chunks,
        })
    return out


# -----------------------------
# Main pipeline
# -----------------------------

def run_pipeline(input_path: str, output_path: str, verbose: bool = True):
    entities, relations = load_input(input_path)
    if verbose:
        print(f'Loaded {len(entities)} entities and {len(relations)} relations')

    # initial enrichment
    enrich_entities(entities)

    # dedupe
    final_entities, id_map, review_candidates = dedupe_entities(entities)
    if verbose:
        print(f'Canonical entities after dedupe: {len(final_entities)}')
        print(f'Review candidates (fuzzy zone): {len(review_candidates)}')

    # update relations
    new_rels = update_relations(relations, id_map, final_entities)

    new_rels = dedupe_relations(new_rels)
    if verbose:
        print(f'Relations after dedupe: {len(new_rels)}')

    # prune relations
    kept_rels, quarantined_rels = prune_relations(new_rels, final_entities)
    if verbose:
        print(f'Kept relations: {len(kept_rels)}, Quarantined: {len(quarantined_rels)}')

    # build output
    out = {
        'entities': to_serializable_entities(final_entities),
        'relations': to_serializable_relations(kept_rels),
        'quarantine': {
            'relations': to_serializable_relations(quarantined_rels),
        },
        'stats': {
            'input_entity_count': len(entities),
            'canonical_entity_count': len(final_entities),
            'input_relation_count': len(relations),
            'kept_relation_count': len(kept_rels),
            'quarantined_relation_count': len(quarantined_rels),
            'review_candidates': len(review_candidates),
        }
    }

    # write files
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    # id_map
    id_map_path = output_path.replace('.json', '.id_map.json')
    with open(id_map_path, 'w', encoding='utf-8') as f:
        json.dump(id_map, f, indent=2, ensure_ascii=False)
    # review candidates
    review_path = output_path.replace('.json', '.review.json')
    with open(review_path, 'w', encoding='utf-8') as f:
        json.dump({'review_candidates': review_candidates}, f, indent=2, ensure_ascii=False)

    print('Written:', output_path)
    print('ID map:', id_map_path)
    print('Review candidates:', review_path)


# -----------------------------
# CLI
# -----------------------------

def post():
    parser = argparse.ArgumentParser(description='KG postprocessing pipeline')
    parser.add_argument('--input', '-i', default="knowledge_graph.json", help='Input JSON (original KG)')
    parser.add_argument('--output', '-o', default='cleaned_output.json', help='Output cleaned JSON')
    parser.add_argument('--no-embeddings', action='store_true', help='Disable embedding-based dedupe')
    args = parser.parse_args()
    global USE_EMBEDDINGS
    if args.no_embeddings:
        USE_EMBEDDINGS = False
    run_pipeline(args.input, args.output)


if __name__ == '__main__':
    post()
