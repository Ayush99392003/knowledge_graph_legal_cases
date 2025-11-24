import re
import hashlib
from typing import List, Dict
import spacy
from fuzzywuzzy import fuzz

# -----------------------------------------------------------
# Load SpaCy NER (English)
# -----------------------------------------------------------
nlp = spacy.load("en_core_web_sm")


# -----------------------------------------------------------
# 1. Base OCR Cleaner
# -----------------------------------------------------------
def clean_ocr(text: str) -> str:
    if not text:
        return ""

    # Remove sequences like ..... ----- _____
    text = re.sub(r"[.\-_/\\]{3,}", " ", text)

    # Remove very long garbage tokens
    text = re.sub(r"\b[a-zA-Z0-9]{25,}\b", " ", text)

    # Fix hyphenation across lines
    text = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", text)

    # Replace unicode spaces and BOM
    for bad, good in {
        "\u00a0": " ",
        "\u200b": "",
        "\u2010": "-",
        "\ufeff": ""
    }.items():
        text = text.replace(bad, good)

    # Remove bullet symbols
    text = re.sub(r"[•●◆■□▪◦►]", " ", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# -----------------------------------------------------------
# 2. Header/Footer Removal
# -----------------------------------------------------------
def remove_repeated_lines(text: str, threshold: int = 3) -> str:
    """
    Remove repeated lines across the whole document.
    This works on the merged document instead of per-page.
    """
    lines = text.split("\n")

    freq = {}
    for line in lines:
        ls = line.strip()
        if len(ls) > 3:
            freq[ls] = freq.get(ls, 0) + 1

    cleaned = []
    for line in lines:
        ls = line.strip()
        if len(ls) > 3 and freq.get(ls, 0) >= threshold:
            continue
        cleaned.append(line)

    return "\n".join(cleaned)


# -----------------------------------------------------------
# 3. Legal Cleaner
# -----------------------------------------------------------
def clean_legal(text: str) -> str:
    if not text:
        return ""

    # Remove "Equivalent citations:" blocks
    text = re.sub(
        r"Equivalent citations:.*?(?=[A-Z][a-z]+|REPORTABLE|JUDGMENT)",
        "",
        text,
        flags=re.DOTALL
    )

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove page numbers on lines alone
    text = re.sub(r"(?m)^\s*\d+\s*$", "", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# -----------------------------------------------------------
# 4. Dedupe (kept for safety)
# -----------------------------------------------------------
def deduplicate(chunks: List[str]) -> List[str]:
    seen = set()
    out = []
    for c in chunks:
        h = hashlib.md5(c.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            out.append(c)
    return out


# -----------------------------------------------------------
# 5. Entity Extraction & Centralization
# -----------------------------------------------------------
def extract_and_centralize_entities(text: str) -> (str, Dict[str, str]):
    doc = nlp(text)
    canonical_map = {}

    # Collect entities
    LEGAL_ENTITY_LABELS = {
    "PERSON",    # judges, individual parties
    "ORG",       # State Transport Authority, corporations
    "GPE",       # State names, locations
    "LOC",       # locations
    "LAW",       # statutes (spaCy sometimes finds)
    "WORK_OF_ART", # sometimes spaCy tags case names
    "NORP",      # groups/communities that may be parties
    "DATE"       # judgment date, helpful metadata
}


    entities = []
    for ent in doc.ents:
        if ent.label_ in LEGAL_ENTITY_LABELS:
            entities.append(ent.text.strip())

    # Centralize similar names
    for ent in entities:
        if len(ent) < 3:
            continue

        found = False
        for canon in canonical_map.values():
            if fuzz.partial_ratio(ent, canon) > 85:
                canonical_map[ent] = canon
                found = True
                break

        if not found:
            canonical_map[ent] = ent

    # Replace variants in text
    for variant, canon in sorted(canonical_map.items(), key=lambda x: -len(x[0])):
        if variant != canon:
            text = re.sub(re.escape(variant), canon, text)

    return text, canonical_map


# -----------------------------------------------------------
# Normalize text with canonical entities
# -----------------------------------------------------------
def normalize_text_for_llm(text: str, canonical_map: dict) -> str:
    for variant, canon in sorted(canonical_map.items(), key=lambda x: -len(x[0])):
        if variant != canon:
            text = re.sub(re.escape(variant), canon, text)

    # Remove repeated patterns like "Court Court"
    text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# -----------------------------------------------------------
# 6. MAIN PIPELINE — ALWAYS RETURN ONE BIG CHUNK
# -----------------------------------------------------------
def processing(chunks: List[Dict]) -> List[Dict]:
    """
    Input: [{"text": "..."}]
    Output:
        [{
            "chunk_id": 0,
            "text": "<fully cleaned single big chunk>",
            "entities": {...}
        }]
    """

    # Merge everything into one big document
    full_text = "\n".join(c.get("text", "") for c in chunks)

    # OCR clean
    full_text = clean_ocr(full_text)

    # Remove headers/footers (document-level)
    full_text = remove_repeated_lines(full_text)

    # Legal cleaning
    full_text = clean_legal(full_text)

    # Entity extraction + canonicalization
    full_text, entity_map = extract_and_centralize_entities(full_text)

    # Final normalization
    full_text = normalize_text_for_llm(full_text, entity_map)

    # One final output chunk
    output_chunks = [full_text]

    output_chunks = deduplicate(output_chunks)

    return [
        {
            "chunk_id": 0,
            "text": output_chunks[0],
            "entities": entity_map
        }
    ]

# import re
# import hashlib
# from typing import List, Dict
# from collections import Counter
# import spacy
# from fuzzywuzzy import fuzz

# # -----------------------------------------------------------
# # Load SpaCy NER (English)
# # -----------------------------------------------------------
# nlp = spacy.load("en_core_web_sm")


# # -----------------------------------------------------------
# # 1. Base OCR Cleaner (SAFE)
# # -----------------------------------------------------------
# def clean_ocr(text: str) -> str:
#     if not text:
#         return ""

#     # Remove sequences like ..... ----- _____
#     text = re.sub(r"[.\-_/\\]{3,}", " ", text)

#     # Remove very long garbage tokens
#     text = re.sub(r"\b[a-zA-Z0-9]{25,}\b", " ", text)

#     # Fix hyphenation across lines
#     text = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", text)

#     # Replace unicode spaces and BOM
#     for bad, good in {
#         "\u00a0": " ",
#         "\u200b": "",
#         "\u2010": "-",
#         "\ufeff": ""
#     }.items():
#         text = text.replace(bad, good)

#     # Remove bullet symbols
#     text = re.sub(r"[•●◆■□▪◦►]", " ", text)

#     # Collapse multiple spaces
#     text = re.sub(r"\s+", " ", text)

#     return text.strip()


# # -----------------------------------------------------------
# # 2. Header/Footer Removal
# # -----------------------------------------------------------
# def remove_repeated_lines(pages: List[str], threshold: int = 3):
#     freq = Counter()
#     for page in pages:
#         for line in page.split("\n"):
#             ls = line.strip()
#             if len(ls) > 3:
#                 freq[ls] += 1
#     cleaned = []
#     for page in pages:
#         out = []
#         for line in page.split("\n"):
#             ls = line.strip()
#             if len(ls) > 3 and freq[ls] < threshold:
#                 out.append(line)
#         cleaned.append("\n".join(out))
#     return cleaned


# # -----------------------------------------------------------
# # 3. Legal Cleaner (SAFE)
# # -----------------------------------------------------------
# def clean_legal(text: str) -> str:
#     if not text:
#         return ""

#     # Remove "Equivalent citations:" blocks
#     text = re.sub(
#         r"Equivalent citations:.*?(?=[A-Z][a-z]+|REPORTABLE|JUDGMENT)",
#         "",
#         text,
#         flags=re.DOTALL
#     )

#     # Remove URLs
#     text = re.sub(r"http\S+", "", text)

#     # Remove page numbers on lines alone
#     text = re.sub(r"(?m)^\s*\d+\s*$", "", text)

#     # Normalize spaces
#     text = re.sub(r"\s+", " ", text)

#     return text.strip()


# # -----------------------------------------------------------
# # 4. Dedupe chunks
# # -----------------------------------------------------------
# def deduplicate(chunks: List[str]) -> List[str]:
#     seen = set()
#     out = []
#     for c in chunks:
#         h = hashlib.md5(c.encode()).hexdigest()
#         if h not in seen:
#             seen.add(h)
#             out.append(c)
#     return out


# # -----------------------------------------------------------
# # 5. Entity Extraction & Centralization
# # -----------------------------------------------------------
# def extract_and_centralize_entities(text: str) -> (str, Dict[str, str]):
#     """
#     Extract PERSON, ORG, COURT, LAW entities and centralize similar names.
#     Returns updated text + mapping {variant: canonical}.
#     """
#     doc = nlp(text)
#     canonical_map = {}

#     # Collect named entities
#     entities = []
#     for ent in doc.ents:
#         if ent.label_ in ("PERSON", "ORG", "LAW", "GPE", "LOC"):
#             entities.append(ent.text.strip())

#     # Centralize using simple fuzzy matching
#     for ent in entities:
#         # Skip very short words
#         if len(ent) < 3:
#             continue
#         found = False
#         for canon in canonical_map.values():
#             if fuzz.partial_ratio(ent, canon) > 85:
#                 canonical_map[ent] = canon
#                 found = True
#                 break
#         if not found:
#             canonical_map[ent] = ent

#     # Replace variants in text
#     # Replace variants in text
#     for variant, canon in sorted(canonical_map.items(), key=lambda x: -len(x[0])):
#         if variant != canon:
#             text = re.sub(re.escape(variant), canon, text)

#     return text, canonical_map

# import re

# def normalize_text_for_llm(text: str, canonical_map: dict) -> str:
#     """
#     Replace all variants in text with canonical entity names.
#     Clean repeated artifacts like 'vs vs' or 'Bench Bench'.
#     """

#     # Step 1: Replace entity variants with canonical names
#     for variant, canon in sorted(canonical_map.items(), key=lambda x: -len(x[0])):
#         if variant != canon:
#             text = re.sub(re.escape(variant), canon, text)

#     # Step 2: Remove repeated words like 'vs vs', 'Bench Bench'
#     text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text)

#     # Step 3: Remove extra spaces
#     text = re.sub(r'\s+', ' ', text)

#     return text.strip()

# # -----------------------------------------------------------
# # 6. Main Pipeline
# # -----------------------------------------------------------
# def processing(chunks: List[Dict]) -> List[Dict]:
#     """
#     Input: [{"text": "page text...", "page": 1}, ...]
#     Output: [{"chunk_id": 0, "text": "...", "entities": {...}}, ...]
#     """

#     # Step A: extract page-wise text
#     pages = [c.get("text", "") for c in chunks]

#     # Step B: OCR cleanup
#     pages = [clean_ocr(p) for p in pages]

#     # Step C: remove repeated headers/footers
#     pages = remove_repeated_lines(pages)

#     # Step D: legal-specific cleanup
#     pages = [clean_legal(p) for p in pages]

#     # Step E: merge all pages
#     merged = "\n".join(pages).strip()

#     # Step F: extract and centralize entities
#     merged, entity_map = extract_and_centralize_entities(merged)
#     for chunk in chunks:
#         chunk["text"] = normalize_text_for_llm(chunk["text"], entity_map)


#     # Step G: split into 800-word chunks
#     words = merged.split()
#     chunk_size = 800
#     chunks_out = [
#         " ".join(words[i:i + chunk_size])
#         for i in range(0, len(words), chunk_size)
#     ]

#     # Step H: deduplicate chunks
#     chunks_out = deduplicate(chunks_out)

#     # Step I: return structured output
#     return [
#         {"chunk_id": i, "text": c, "entities": entity_map}
#         for i, c in enumerate(chunks_out)
#     ]

# import re
# import hashlib
# from typing import List, Dict
# from collections import Counter


# # -----------------------------------------------------------
# # 1. Base OCR Cleaner (SAFE)
# # -----------------------------------------------------------

# def clean_ocr(text: str) -> str:
#     if not text:
#         return ""

#     # Remove ..... ----- _____
#     text = re.sub(r"[.\-_/\\]{3,}", " ", text)

#     # Remove garbage tokens like: ASDJJSD933DKDKEWWKDK
#     text = re.sub(r"\b[a-zA-Z0-9]{25,}\b", " ", text)

#     # Fix hyphenation (only when broken across lines)
#     text = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", text)

#     # Replace unicode spaces
#     for bad, good in {
#         "\u00a0": " ",
#         "\u200b": "",
#         "\u2010": "-",
#         "\ufeff": ""
#     }.items():
#         text = text.replace(bad, good)

#     # Remove bullet symbols
#     text = re.sub(r"[•●◆■□▪◦►]", " ", text)

#     # Collapse space
#     text = re.sub(r"\s+", " ", text)

#     return text.strip()


# # -----------------------------------------------------------
# # 2. Header/Footer Removal (Safe)
# # -----------------------------------------------------------

# def remove_repeated_lines(pages: List[str], threshold: int = 3):
#     freq = Counter()

#     for page in pages:
#         for line in page.split("\n"):
#             ls = line.strip()
#             if len(ls) > 3:
#                 freq[ls] += 1

#     cleaned = []
#     for page in pages:
#         out = []
#         for line in page.split("\n"):
#             ls = line.strip()
#             if len(ls) > 3 and freq[ls] < threshold:
#                 out.append(line)
#         cleaned.append("\n".join(out))

#     return cleaned


# # -----------------------------------------------------------
# # 3. Legal Cleaner (SAFE)
# # -----------------------------------------------------------

# def clean_legal(text: str) -> str:
#     if not text:
#         return ""

#     # Remove "Equivalent citations:" full block
#     text = re.sub(
#         r"Equivalent citations:.*?(?=[A-Z][a-z]+|REPORTABLE|JUDGMENT)",
#         "",
#         text,
#         flags=re.DOTALL
#     )

#     # Remove URLs (like Indian Kanoon)
#     text = re.sub(r"http\S+", "", text)

#     # Remove pure page numbers only if ALONE on a line
#     text = re.sub(r"(?m)^\s*\d+\s*$", "", text)

#     # KEEP paragraph numbers — DO NOT delete "1.", "(2)", etc.
#     # so we DO NOT use your previous destructive regex.

#     # Normalize space
#     text = re.sub(r"\s+", " ", text)

#     return text.strip()


# # -----------------------------------------------------------
# # 4. Dedupe
# # -----------------------------------------------------------

# def deduplicate(chunks: List[str]) -> List[str]:
#     seen = set()
#     out = []
#     for c in chunks:
#         h = hashlib.md5(c.encode()).hexdigest()
#         if h not in seen:
#             seen.add(h)
#             out.append(c)
#     return out


# # -----------------------------------------------------------
# # 5. Main Pipeline (Safe for legal cases)
# # -----------------------------------------------------------

# def processing(chunks: List[Dict]) -> List[Dict]:

#     # Step A — extract page-wise text
#     pages = [c.get("text", "") for c in chunks]

#     # Step B — clean OCR
#     pages = [clean_ocr(p) for p in pages]

#     # Step C — remove repeated headers/footers
#     pages = remove_repeated_lines(pages)

#     # Step D — legal cleanup (safe)
#     pages = [clean_legal(p) for p in pages]

#     # Step E — merge all pages
#     merged = "\n".join(pages).strip()

#     # Step F — chunk by words (800 tokens)
#     words = merged.split()
#     chunk_size = 800
#     chunks_out = [
#         " ".join(words[i:i + chunk_size])
#         for i in range(0, len(words), chunk_size)
#     ]

#     # Step G — dedupe
#     chunks_out = deduplicate(chunks_out)

#     # Build final output
#     return [{"chunk_id": i, "text": c} for i, c in enumerate(chunks_out)]


# import re
# from typing import List, Dict

# def clean_ocr(text: str) -> str:
#     if not text:
#         return ""

#     # -------------------------------------------------------
#     # 1. Remove common OCR artifacts: ".....", "-----", "_____"
#     # -------------------------------------------------------
#     text = re.sub(r'[.\-_/\\]{3,}', ' ', text)

#     # -------------------------------------------------------
#     # 2. Remove long garbage tokens caused by OCR
#     # (e.g. "ODXJTLLS94494KDOWW" etc.)
#     # -------------------------------------------------------
#     text = re.sub(r'\b[a-zA-Z0-9]{20,}\b', ' ', text)

#     # -------------------------------------------------------
#     # 3. Fix hyphenated words split across lines
#     # "constitu- tion" → "constitution"
#     # -------------------------------------------------------
#     text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)

#     # -------------------------------------------------------
#     # 4. Remove stray page markers like "[p1 1]" or "[p23]"
#     # -------------------------------------------------------
#     text = re.sub(r'\[p\d+\s*\d*\]', ' ', text)
#     text = re.sub(r'\[p\d+\]', ' ', text)

#     # -------------------------------------------------------
#     # 5. Normalize weird unicode spaces & line breaks
#     # -------------------------------------------------------
#     text = text.replace("\u00a0", " ")     # NBSP
#     text = text.replace("\u200b", "")      # zero-width space
#     text = text.replace("\u2010", "-")     # unicode hyphen

#     # -------------------------------------------------------
#     # 6. Remove multiple spaces
#     # -------------------------------------------------------
#     text = re.sub(r'\s+', ' ', text)

#     return text.strip()


# def processing(chunks: List[Dict]) -> List[Dict]:
#     """
#     Adds a clean_text field for each chunk.
#     """
#     for entry in chunks:
#         raw = entry.get("text", "")
#         entry["text"] = clean_ocr(raw)

#     return chunks


# import json
# import re
# import nltk
# from typing import List, Dict
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer, PorterStemmer
# from nltk.tokenize import word_tokenize
# from typing import List, Dict
# def downloads_nltk():
#     nltk.download('punkt')
#     nltk.download('stopwords')
#     nltk.download('wordnet')
# def processing(chunks: List[Dict]) -> List[Dict]:
#     downloads_nltk()
#     stop_words = set(stopwords.words('english'))
#     lemmatizer = WordNetLemmatizer()
#     stemmer = PorterStemmer()
#     def clean_ocr_artifacts(text: str) -> str:
#         if not text:
#             return ""
#         text = re.sub(r'[\.\-]{3,}', ' ', text)
#         text = re.sub(r'[a-zA-Z0-9]{1,}[a-zA-Z0-9\s]{0,}$', '', text)
#         text = re.sub(r'([a-zA-Z0-9]){15,}', '', text)
#         text = re.sub(r'\s{2,}', ' ', text).strip()
#         return text
#     def preprocess_text(text: str) -> str:
#         text = clean_ocr_artifacts(text)
#         text = text.lower()
#         text = re.sub(r'[^a-z\s]', ' ', text) 
#         tokens = word_tokenize(text)
#         tokens = [word for word in tokens if word not in stop_words]
#         # tokens = [stemmer.stem(lemmatizer.lemmatize(word)) for word in tokens]
#         return ' '.join(tokens)
#     for entry in chunks:
#         if "text" in entry:  
#             entry["text"] = preprocess_text(entry["text"])
#     return chunks