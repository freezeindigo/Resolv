#!/usr/bin/env python3
"""
Build data/rag_index.json from markdown under data/rag_documents/<collection>/*.md

Usage:
  python3 scripts/ingest_rag_documents.py
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC_ROOT = ROOT / "data" / "rag_documents"
OUT = ROOT / "data" / "rag_index.json"

MIN_CHUNK = 40


def chunk_text(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n+", text.strip())
    out = []
    for p in parts:
        p = p.strip()
        if len(p) >= MIN_CHUNK:
            out.append(p)
    if not out and text.strip():
        out = [text.strip()[:4000]]
    return out


def main():
    if not DOC_ROOT.is_dir():
        DOC_ROOT.mkdir(parents=True, exist_ok=True)
        print(f"Created {DOC_ROOT}; add markdown files under data/rag_documents/<collection>/")
        return

    chunks_out = []
    n_docs = 0
    collections: set[str] = set()

    for coll_dir in sorted(DOC_ROOT.iterdir()):
        if not coll_dir.is_dir():
            continue
        collection = coll_dir.name
        collections.add(collection)
        for md in sorted(coll_dir.glob("*.md")):
            n_docs += 1
            title = md.stem.replace("_", " ").title()
            raw = md.read_text(encoding="utf-8", errors="replace")
            for i, ch in enumerate(chunk_text(raw)):
                cid = hashlib.sha256(f"{collection}:{md.name}:{i}:{ch[:80]}".encode()).hexdigest()[:16]
                chunks_out.append(
                    {
                        "id": cid,
                        "collection": collection,
                        "title": title,
                        "source_file": str(md.relative_to(ROOT)),
                        "text": ch,
                    }
                )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "collections": sorted(collections),
        "documents": n_docs,
        "chunks": len(chunks_out),
        "chunks_data": chunks_out,
    }
    # Flatten for runtime retriever: chunks key
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(
            {
                "collections": payload["collections"],
                "documents": n_docs,
                "chunks": chunks_out,
            },
            f,
            indent=2,
        )

    print(
        f"Ingest complete: {n_docs} documents, {len(chunks_out)} chunks "
        f"across {len(collections)} collections"
    )
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
