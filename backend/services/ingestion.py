"""Extracción y troceo de texto desde texto plano, URL y PDF."""
from io import BytesIO
from typing import List, Tuple
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
    text = text.strip()
    if not text:
        return []
    if chunk_size <= overlap:
        raise ValueError("chunk_size debe ser mayor que overlap")
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - overlap
    return chunks


def chunk_pages(
    pages: List[str],
    chunk_size: int = 1000,
    overlap: int = 150,
) -> List[Tuple[str, int]]:
    """Trocea respetando los límites de página: cada chunk lleva su número de página real (1-indexed)."""
    out: List[Tuple[str, int]] = []
    for idx, raw in enumerate(pages, start=1):
        page_text = (raw or "").strip()
        if not page_text:
            continue
        for piece in chunk_text(page_text, chunk_size, overlap):
            out.append((piece, idx))
    return out


def extract_url(url: str, timeout: int = 15) -> tuple[str, str]:
    """Devuelve (título, texto_limpio) de una URL."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("URL inválida; debe empezar con http(s)://")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; AnswerFactoryLab/1.0; "
            "+https://crash-course-ia-docentes.local)"
        )
    }
    resp = requests.get(url, timeout=timeout, headers=headers)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    for tag in soup(["script", "style", "noscript", "iframe", "header", "footer", "nav", "aside"]):
        tag.decompose()

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    if not title:
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(strip=True)
    if not title:
        title = parsed.netloc

    text = soup.get_text(separator="\n", strip=True)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    cleaned = "\n".join(lines)
    return title, cleaned


def extract_pdf(file_bytes: bytes) -> str:
    if not file_bytes:
        return ""
    text = extract_text(BytesIO(file_bytes)) or ""
    return text.strip()


def extract_pdf_pages(file_bytes: bytes) -> List[str]:
    """Devuelve el texto del PDF página por página (lista 1-indexada por su orden)."""
    if not file_bytes:
        return []
    full = extract_text(BytesIO(file_bytes)) or ""
    if not full:
        return []
    # pdfminer separa páginas con form-feed (\f) por default.
    pages = full.split("\f")
    # La última página suele venir vacía por el \f final.
    while pages and not pages[-1].strip():
        pages.pop()
    return pages
