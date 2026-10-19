"""Local, page-aware retrieval for PDFs in ``documents/``."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import faiss
import numpy as np
from pypdf import PdfReader

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-m3")


@dataclass(frozen=True)
class RagIndex:
    """Embedding model, vector index, and chunks carrying their citations."""

    model: Any
    index: faiss.Index
    chunks: list[dict[str, object]]
    fingerprint: str


# 1. EXTRACT TEXT FROM PDFs

def pdf_fingerprint(folder: str | Path) -> str:
    """A cache key which changes whenever an indexed PDF changes."""
    parts = []
    for path in sorted(Path(folder).glob("*.pdf"), key=lambda item: item.name.lower()):
        stat = path.stat()
        parts.append(f"{path.name}:{stat.st_mtime_ns}:{stat.st_size}")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def extract_text_from_pdf(pdf_path: str | Path) -> list[dict[str, object]]:
    """
    Extract text from all pages of one PDF.
    """

    reader = PdfReader(str(pdf_path))

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()

        if text:
            pages.append({
                "page": page_number,
                "text": text
            })

    return pages



# 2. LOAD ALL DOCUMENTS

def load_documents(folder: str | Path = "documents") -> list[dict[str, object]]:
    """
    Load every PDF in the documents folder.
    """

    documents = []

    for pdf_path in sorted(Path(folder).glob("*.pdf"), key=lambda item: item.name.lower()):

        pages = extract_text_from_pdf(pdf_path)

        for page in pages:

            documents.append({
                "source": pdf_path.name,
                "page": page["page"],
                "text": page["text"]
            })

    return documents


# 3. CHUNK TEXT

def chunk_text(text: str, chunk_size: int = 220, overlap: int = 40) -> list[str]:
    """
    Split text into overlapping chunks.
    """

    words = text.split()
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        if chunk := " ".join(words[start:end]).strip():
            chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap

    return chunks


def create_chunks(documents: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    """
    Create chunks while keeping source + page metadata.
    """

    chunks = []

    for document in documents:

        text_chunks = chunk_text(document["text"])

        for chunk_number, chunk in enumerate(text_chunks, start=1):

            chunks.append({
                "text": chunk,
                "source": document["source"],
                "page": document["page"],
                "chunk": chunk_number,
            })

    return chunks



# 4. CREATE EMBEDDINGS


def create_embeddings(chunks: list[dict[str, object]], model_name: str = DEFAULT_EMBEDDING_MODEL) -> tuple[Any, np.ndarray]:
    """
    Convert chunks into vectors.
    """

    if not chunks:
        raise ValueError("No extractable PDF text was found in the documents folder.")
    # Importing torch-backed encoders is slow; do it only when PDFs actually
    # need indexing so the regular review UI remains fast.
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return model, np.asarray(embeddings, dtype="float32")



# 5. CREATE VECTOR DATABASE


def create_vector_db(embeddings: np.ndarray) -> faiss.Index:
    """
    Store embeddings inside FAISS.
    """

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(
        np.asarray(embeddings, dtype="float32")
    )

    return index



# 6. RETRIEVE RELEVANT CHUNKS


def retrieve(question: str, rag: RagIndex, top_k: int = 6) -> list[dict[str, object]]:
    """
    Find the most relevant chunks for a question.
    """

    if not question.strip():
        return []
    candidate_count = min(len(rag.chunks), max(top_k * 4, top_k))
    query = rag.model.encode([question], normalize_embeddings=True)
    scores, indices = rag.index.search(np.asarray(query, dtype="float32"), candidate_count)
    results: list[dict[str, object]] = []
    per_page: dict[tuple[str, int], int] = {}
    for score, index_id in zip(scores[0], indices[0]):
        if index_id < 0:
            continue
        item = rag.chunks[int(index_id)]
        page_key = (str(item["source"]), int(item["page"]))
        if per_page.get(page_key, 0) >= 2:
            continue
        chunk = dict(item)
        chunk["score"] = float(score)
        results.append(chunk)
        per_page[page_key] = per_page.get(page_key, 0) + 1
        if len(results) == top_k:
            break

    return results



# 7. BUILD RAG PROMPT


def build_prompt(question: str, retrieved_chunks: list[dict[str, object]]) -> str:

    context = "\n\n".join(
        f"[Source {position}: {chunk['source']}, page {chunk['page']}]\n{chunk['text']}"
        for position, chunk in enumerate(retrieved_chunks, start=1)
    )

    prompt = f"""
Answer the user's question using only the supplied PDF excerpts.
If the excerpts do not contain the answer, say so plainly. Do not infer or invent facts.
Reply in the user's language. Cite supporting excerpts inline using [Source N].

PDF EXCERPTS:
{context}

USER QUESTION:
{question}
"""

    return prompt



# 8. COMPLETE RAG PIPELINE

def build_rag(folder: str | Path = "documents", model_name: str = DEFAULT_EMBEDDING_MODEL) -> RagIndex:
    """Build an in-memory index. Cache this function at the UI boundary."""
    documents = load_documents(folder)
    chunks = create_chunks(documents)
    model, embeddings = create_embeddings(chunks, model_name)
    return RagIndex(model=model, index=create_vector_db(embeddings), chunks=chunks, fingerprint=pdf_fingerprint(folder))


def references_from_chunks(chunks: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    """Collapse snippets into unique PDF/page references for the chat UI."""
    references: list[dict[str, object]] = []
    seen: set[tuple[str, int]] = set()
    for chunk in chunks:
        key = (str(chunk["source"]), int(chunk["page"]))
        if key not in seen:
            references.append({"source": key[0], "page": key[1], "score": float(chunk["score"])})
            seen.add(key)
    return references


# 9. TEST 

if __name__ == "__main__":

    rag = build_rag()

    while True:

        question = input("\nQuestion: ")

        if question.lower() == "exit":
            break

        results = retrieve(
            question,
            rag,
            top_k=5
        )

        print("\n--- Retrieved Context ---")

        for result in results:

            print(
                f"\n[{result['source']} - "
                f"page {result['page']}] "
                f"score={result['score']:.3f}"
            )

            print(result["text"][:500])

        prompt = build_prompt(
            question,
            results
        )

        print("\n--- PROMPT FOR QWEN ---")
        print(prompt)
