"""
rag.py — Knowledge-base engine for StatVerse.

Upload flow (runs in a background thread so large books never time out):
  extract text page-by-page -> chunk (page-aware) -> embed in small batches
  -> store in local ChromaDB.  Progress is reported through a job object
  that the browser polls.

Embeddings are local + free (sentence-transformers). Nothing here calls a paid API.
"""

import bisect
import os
import re
import threading
import time
import uuid

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "statverse_books"
EMBED_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 64
# Chunks farther than this from the query are treated as "not really about this topic".
MAX_RELEVANT_DISTANCE = 1.25

_lock = threading.Lock()
_collection = None


def _get_collection():
    """Lazy init so the server starts instantly and model-load errors are reportable."""
    global _collection
    if _collection is None:
        with _lock:
            if _collection is None:
                fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
                client = chromadb.PersistentClient(path=CHROMA_DIR)
                _collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=fn)
    return _collection


# ───────────────────────── text extraction & chunking ─────────────────────────

def _clean(t: str) -> str:
    t = t.replace("\x00", " ")
    return re.sub(r"\s+", " ", t).strip()


def extract_pages(file_path: str, on_progress=None) -> list[str]:
    """Return a list of cleaned page texts (a .txt file is treated as one page)."""
    if file_path.lower().endswith(".pdf"):
        reader = PdfReader(file_path)
        total = len(reader.pages)
        pages = []
        for i, page in enumerate(reader.pages):
            try:
                pages.append(_clean(page.extract_text() or ""))
            except Exception:
                pages.append("")
            if on_progress and (i % 5 == 0 or i == total - 1):
                on_progress(i + 1, total)
        return pages
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return [_clean(f.read())]


def chunk_pages(pages: list[str], chunk_size: int = 900, overlap: int = 150) -> list[tuple[str, int]]:
    """Chunk the whole book but remember which page each chunk starts on."""
    text_parts, starts = [], []
    offset = 0
    for pno, ptxt in enumerate(pages, start=1):
        if not ptxt:
            continue
        starts.append((offset, pno))
        text_parts.append(ptxt)
        offset += len(ptxt) + 1
    text = " ".join(text_parts)
    if not text:
        return []
    offsets = [s[0] for s in starts]

    def page_at(pos: int) -> int:
        return starts[max(bisect.bisect_right(offsets, pos) - 1, 0)][1]

    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            b = text.rfind(". ", start, end)
            if b != -1 and b > start + 200:
                end = b + 1
        piece = text[start:end].strip()
        if len(piece) > 40:
            chunks.append((piece, page_at(start)))
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


# ───────────────────────────── background indexing ─────────────────────────────

JOBS: dict[str, dict] = {}


def start_index_job(file_path: str, book_title: str, topic_tag: str = "") -> str:
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {"status": "running", "progress": 1, "message": "Starting…", "chunks": 0, "error": None, "book": book_title}
    threading.Thread(target=_run_job, args=(job_id, file_path, book_title, topic_tag), daemon=True).start()
    return job_id


def _run_job(job_id, file_path, book_title, topic_tag):
    job = JOBS[job_id]
    try:
        def read_prog(done, total):
            job["progress"] = 1 + int(29 * done / max(total, 1))
            job["message"] = f"Reading page {done} of {total}…"

        pages = extract_pages(file_path, read_prog)
        chunks = chunk_pages(pages)
        if not chunks:
            raise ValueError(
                "No selectable text found in this file. It looks like a scanned/photo PDF. "
                "Scanned pages need OCR first (e.g. run it through ocrmypdf), then upload again."
            )

        job["message"] = "Loading embedding model (first time takes a minute)…"
        col = _get_collection()

        with _lock:  # re-uploading the same book replaces the old copy
            col.delete(where={"book_title": book_title})

        total = len(chunks)
        for i in range(0, total, BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            with _lock:
                col.add(
                    documents=[c[0] for c in batch],
                    ids=[str(uuid.uuid4()) for _ in batch],
                    metadatas=[{"book_title": book_title, "topic_tag": topic_tag, "page": c[1], "chunk_index": i + j}
                               for j, c in enumerate(batch)],
                )
            done = min(i + BATCH_SIZE, total)
            job["progress"] = 30 + int(69 * done / total)
            job["message"] = f"Indexing {done} of {total} sections…"

        job.update(status="done", progress=100, chunks=total, message=f"Done — {total} sections indexed.")
    except Exception as e:  # surfaced to the UI
        job.update(status="error", error=str(e), message="Failed")


# ───────────────────────────────── retrieval ─────────────────────────────────

def retrieve(query: str, n_results: int = 6) -> list[dict]:
    col = _get_collection()
    count = col.count()
    if count == 0:
        return []
    res = col.query(query_texts=[query], n_results=min(n_results, count))
    out = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        out.append({
            "text": doc,
            "book_title": meta.get("book_title", "Unknown"),
            "page": meta.get("page"),
            "distance": float(dist),
            "relevant": float(dist) <= MAX_RELEVANT_DISTANCE,
        })
    return out


def library() -> dict:
    col = _get_collection()
    if col.count() == 0:
        return {"total_chunks": 0, "books": []}
    counts: dict[str, int] = {}
    for m in col.get(include=["metadatas"])["metadatas"]:
        t = m.get("book_title", "Unknown")
        counts[t] = counts.get(t, 0) + 1
    return {"total_chunks": col.count(),
            "books": [{"title": t, "chunks": c} for t, c in sorted(counts.items())]}


def delete_book(title: str):
    with _lock:
        _get_collection().delete(where={"book_title": title})
