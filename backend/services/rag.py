"""Capa RAG sobre ChromaDB persistente, scoped por sesión."""
from datetime import datetime
from typing import Iterable, List, Dict, Any
import uuid

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from config import (
    CHROMA_DB_PATH,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
)

_client = None
_embedding_fn = None


def _get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=str(CHROMA_DB_PATH),
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def _get_embedding_fn():
    global _embedding_fn
    if _embedding_fn is None:
        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY es requerida para los embeddings del RAG"
            )
        _embedding_fn = OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name=EMBEDDING_MODEL,
        )
    return _embedding_fn


def _collection_name(session_id: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)
    return f"docente_{safe}"[:60]


def _get_collection(session_id: str, *, require_embeddings: bool = True):
    """Devuelve la colección de la sesión.

    Cuando ``require_embeddings`` es True (queries / writes) cargamos la
    embedding function. Para operaciones de solo lectura/borrado (status,
    clear) lo evitamos para que el endpoint funcione aun sin OPENAI_API_KEY.
    """
    kwargs = {
        "name": _collection_name(session_id),
        "metadata": {"hnsw:space": "cosine"},
    }
    if require_embeddings:
        kwargs["embedding_function"] = _get_embedding_fn()
    return _get_client().get_or_create_collection(**kwargs)


def add_chunks(
    session_id: str,
    chunks: List[str],
    *,
    title: str,
    author: str,
    source_type: str,
    source_ref: str,
) -> int:
    if not chunks:
        return 0
    collection = _get_collection(session_id)
    now = datetime.utcnow().isoformat()
    doc_id = str(uuid.uuid4())[:8]
    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "title": title or "Sin título",
            "author": author or "Autor desconocido",
            "source_type": source_type,
            "source_ref": source_ref,
            "page": i + 1,
            "chunk_index": i,
            "uploaded_at": now,
        }
        for i in range(len(chunks))
    ]
    collection.add(documents=chunks, metadatas=metadatas, ids=ids)
    return len(chunks)


def query(session_id: str, prompt: str, n_results: int = 4) -> Dict[str, Any]:
    collection = _get_collection(session_id)
    if collection.count() == 0:
        return {"chunks": [], "metadatas": [], "distances": []}
    res = collection.query(query_texts=[prompt], n_results=min(n_results, collection.count()))
    return {
        "chunks": res.get("documents", [[]])[0],
        "metadatas": res.get("metadatas", [[]])[0],
        "distances": res.get("distances", [[]])[0],
    }


def status(session_id: str) -> Dict[str, Any]:
    collection = _get_collection(session_id, require_embeddings=False)
    count = collection.count()
    if count == 0:
        return {"total_chunks": 0, "sources": []}

    sample = collection.get(limit=min(count, 200), include=["metadatas"])
    seen: Dict[str, Dict[str, Any]] = {}
    for meta in sample.get("metadatas") or []:
        key = f"{meta.get('source_type')}::{meta.get('source_ref')}::{meta.get('title')}"
        if key not in seen:
            seen[key] = {
                "title": meta.get("title"),
                "author": meta.get("author"),
                "type": meta.get("source_type"),
                "source_ref": meta.get("source_ref"),
                "uploaded_at": meta.get("uploaded_at"),
            }
    return {"total_chunks": count, "sources": list(seen.values())}


def clear(session_id: str) -> None:
    name = _collection_name(session_id)
    try:
        _get_client().delete_collection(name=name)
    except Exception:
        pass


def format_context_blocks(retrieved: Dict[str, Any]) -> List[str]:
    chunks = retrieved.get("chunks") or []
    metas = retrieved.get("metadatas") or []
    blocks = []
    for chunk, meta in zip(chunks, metas):
        title = meta.get("title", "Sin título")
        author = meta.get("author", "Autor desconocido")
        page = meta.get("page", "?")
        blocks.append(f"[Fuente: {title} — {author} — fragmento {page}]\n{chunk}")
    return blocks


def format_sources(retrieved: Dict[str, Any]) -> List[Dict[str, Any]]:
    metas = retrieved.get("metadatas") or []
    distances = retrieved.get("distances") or []
    out = []
    for meta, dist in zip(metas, distances):
        out.append({
            "title": meta.get("title", "Sin título"),
            "author": meta.get("author", "Autor desconocido"),
            "page": meta.get("page", "?"),
            "type": meta.get("source_type"),
            "similarity": round(1 - float(dist), 3) if dist is not None else None,
        })
    return out
