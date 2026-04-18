"""
Lightweight RAG retrieval (keyword overlap, no embeddings) for operational documents.
Index built by scripts/ingest_rag_documents.py → data/rag_index.json
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
INDEX_PATH = ROOT / "data" / "rag_index.json"

COLLECTION_AUDIT = "audit"
COLLECTION_MOM = "mom"


def _tokenize(text: str) -> set:
    return {t for t in re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", (text or "").lower()) if len(t) > 2}


def _score(query: str, chunk_text: str) -> float:
    q = _tokenize(query)
    c = _tokenize(chunk_text)
    if not q or not c:
        return 0.0
    return len(q & c) / (len(q) ** 0.5)


def load_chunks() -> List[Dict[str, Any]]:
    if not INDEX_PATH.exists():
        return []
    with open(INDEX_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return list(data.get("chunks", []))


def retrieve_for_complaint(
    complaint_title: str,
    site_name: str,
    domain: str,
    top_k: int = 6,
) -> Tuple[List[Dict[str, Any]], List[str], List[str], int]:
    """
    Returns (rag_sources_used, audit_context_snippets, mom_context_snippets, elapsed_ms).
    rag_sources_used entries: collection, title, snippet, score
    """
    t0 = time.monotonic()
    chunks = load_chunks()
    if not chunks:
        return [], [], [], 0

    query = f"{complaint_title} {site_name} {domain}".strip()
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for ch in chunks:
        s = _score(query, ch.get("text", ""))
        if s > 0:
            scored.append((s, ch))
    scored.sort(key=lambda x: -x[0])
    top = scored[:top_k]

    sources: List[Dict[str, Any]] = []
    audit_snips: List[str] = []
    mom_snips: List[str] = []
    for s, ch in top:
        coll = ch.get("collection", "")
        title = ch.get("title", "")
        text = ch.get("text", "")
        snip = text[:320].replace("\n", " ") + ("…" if len(text) > 320 else "")
        sources.append(
            {
                "collection": coll,
                "title": title,
                "snippet": snip,
                "score": round(s, 4),
            }
        )
        if coll == COLLECTION_AUDIT:
            audit_snips.append(snip)
        elif coll == COLLECTION_MOM:
            mom_snips.append(snip)

    elapsed = int((time.monotonic() - t0) * 1000)
    return sources, audit_snips, mom_snips, elapsed
