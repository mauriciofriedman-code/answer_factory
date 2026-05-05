"""The Answer Factory — backend FastAPI."""
import asyncio
import secrets
import logging
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import (
    ALLOWED_ORIGINS,
    DEFAULT_MODEL,
    OPENAI_API_KEY,
    SESSION_HEADER,
    STYLE_PROMPTS,
    SUPPORTED_MODELS,
)
from services import ingestion, llm, rag, tokens, verifier

logger = logging.getLogger("answer_factory")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="The Answer Factory API",
    description="Laboratorio de prompts y parámetros de IA para docentes.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=[SESSION_HEADER],
)


def get_session_id(request: Request, response: Response) -> str:
    sid = request.headers.get(SESSION_HEADER)
    if not sid or len(sid) < 8 or len(sid) > 128:
        sid = secrets.token_urlsafe(16)
    response.headers[SESSION_HEADER] = sid
    return sid


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    model: str = DEFAULT_MODEL
    temperature: float = 0.7
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    max_tokens: int = Field(2000, ge=1, le=4000)
    style: str = "natural"
    use_rag: bool = False
    stop_sequences: List[str] = []
    rag_top_k: int = Field(4, ge=1, le=10)
    return_logprobs: bool = False
    top_logprobs: int = Field(3, ge=1, le=5)


class TokenizeRequest(BaseModel):
    text: str = Field(..., min_length=1)
    model: str = DEFAULT_MODEL


class VerifyRequest(BaseModel):
    response_text: str = Field(..., min_length=1)
    judge_model: str = DEFAULT_MODEL


class VariantRequest(BaseModel):
    label: str = "Variante"
    model: str = DEFAULT_MODEL
    style: str = "natural"
    temperature: float = 0.7
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    max_tokens: int = Field(2000, ge=1, le=4000)
    stop_sequences: List[str] = []


class CompareRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    use_rag: bool = False
    rag_top_k: int = Field(4, ge=1, le=10)
    variants: List[VariantRequest] = Field(..., min_length=2, max_length=4)


class TextUploadRequest(BaseModel):
    text: str = Field(..., min_length=1)
    title: str = "Documento sin título"
    author: str = "Autor desconocido"
    chunk_size: int = Field(1000, ge=200, le=4000)
    chunk_overlap: int = Field(150, ge=0, le=1000)


class URLUploadRequest(BaseModel):
    url: str = Field(..., min_length=4)
    chunk_size: int = Field(1000, ge=200, le=4000)
    chunk_overlap: int = Field(150, ge=0, le=1000)


@app.get("/")
def root():
    return {"message": "The Answer Factory API · v2", "status": "ok"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "openai_configured": bool(OPENAI_API_KEY),
        "models": list(SUPPORTED_MODELS.keys()),
    }


@app.get("/api/config")
def get_public_config():
    return {
        "models": [
            {"id": k, "label": v["label"], "provider": v["provider"]}
            for k, v in SUPPORTED_MODELS.items()
        ],
        "default_model": DEFAULT_MODEL,
        "styles": list(STYLE_PROMPTS.keys()),
    }


@app.post("/api/generate")
def generate_response(payload: GenerateRequest, request: Request, response: Response):
    session_id = get_session_id(request, response)

    if not OPENAI_API_KEY:
        return {
            "response": (
                "[Modo de prueba — OPENAI_API_KEY no configurada]\n"
                f"Prompt recibido: {payload.prompt}"
            ),
            "model": payload.model,
            "chunks_used": 0,
            "sources": [],
            "usage": None,
        }

    context_blocks: List[str] = []
    sources: List[dict] = []
    if payload.use_rag:
        retrieved = rag.query(session_id, payload.prompt, n_results=payload.rag_top_k)
        context_blocks = rag.format_context_blocks(retrieved)
        sources = rag.format_sources(retrieved)

    try:
        result = llm.generate(
            model=payload.model,
            prompt=payload.prompt,
            style=payload.style,
            temperature=payload.temperature,
            top_p=payload.top_p,
            frequency_penalty=payload.frequency_penalty,
            presence_penalty=payload.presence_penalty,
            max_tokens=payload.max_tokens,
            stop_sequences=payload.stop_sequences or None,
            context_blocks=context_blocks,
            return_logprobs=payload.return_logprobs,
            top_logprobs=payload.top_logprobs,
        )
    except Exception as exc:
        logger.exception("generate failed")
        raise HTTPException(status_code=500, detail=f"Error al generar: {exc}")

    return {
        "response": result["text"],
        "model": result["model"],
        "finish_reason": result.get("finish_reason"),
        "chunks_used": len(context_blocks),
        "sources": sources,
        "usage": {
            "input_tokens": result.get("input_tokens"),
            "output_tokens": result.get("output_tokens"),
        },
        "logprobs": result.get("logprobs"),
    }


@app.post("/api/tokenize")
def tokenize_endpoint(payload: TokenizeRequest):
    return tokens.tokenize(payload.text, payload.model)


@app.post("/api/verify-claims")
def verify_claims(payload: VerifyRequest, request: Request, response: Response):
    session_id = get_session_id(request, response)
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY es requerida.")
    if rag.status(session_id)["total_chunks"] == 0:
        raise HTTPException(
            status_code=400,
            detail="Sube al menos una fuente al RAG antes de verificar afirmaciones.",
        )
    try:
        return verifier.verify(
            payload.response_text,
            session_id=session_id,
            model=payload.judge_model,
        )
    except Exception as exc:
        logger.exception("verify failed")
        raise HTTPException(status_code=500, detail=f"Error al verificar: {exc}")


@app.post("/api/compare")
async def compare(payload: CompareRequest, request: Request, response: Response):
    session_id = get_session_id(request, response)

    if not OPENAI_API_KEY:
        return {
            "results": [
                {
                    "label": v.label,
                    "response": "[Modo de prueba — OPENAI_API_KEY no configurada]",
                    "model": v.model,
                    "chunks_used": 0,
                    "sources": [],
                    "usage": None,
                    "error": None,
                }
                for v in payload.variants
            ]
        }

    context_blocks: List[str] = []
    sources: List[dict] = []
    if payload.use_rag:
        retrieved = rag.query(session_id, payload.prompt, n_results=payload.rag_top_k)
        context_blocks = rag.format_context_blocks(retrieved)
        sources = rag.format_sources(retrieved)

    def _run(variant: VariantRequest):
        try:
            r = llm.generate(
                model=variant.model,
                prompt=payload.prompt,
                style=variant.style,
                temperature=variant.temperature,
                top_p=variant.top_p,
                frequency_penalty=variant.frequency_penalty,
                presence_penalty=variant.presence_penalty,
                max_tokens=variant.max_tokens,
                stop_sequences=variant.stop_sequences or None,
                context_blocks=context_blocks,
            )
            return {
                "label": variant.label,
                "response": r["text"],
                "model": r["model"],
                "finish_reason": r.get("finish_reason"),
                "chunks_used": len(context_blocks),
                "sources": sources,
                "usage": {
                    "input_tokens": r.get("input_tokens"),
                    "output_tokens": r.get("output_tokens"),
                },
                "error": None,
            }
        except Exception as exc:
            logger.exception("compare variant failed")
            return {
                "label": variant.label,
                "response": "",
                "model": variant.model,
                "chunks_used": len(context_blocks),
                "sources": sources,
                "usage": None,
                "error": str(exc),
            }

    results = await asyncio.gather(
        *[asyncio.to_thread(_run, v) for v in payload.variants]
    )
    return {"results": results}


def _safe_add_chunks(session_id, chunks, **meta):
    try:
        return rag.add_chunks(session_id, chunks, **meta)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/api/upload-text")
def upload_text(payload: TextUploadRequest, request: Request, response: Response):
    session_id = get_session_id(request, response)
    chunks = ingestion.chunk_text(payload.text, payload.chunk_size, payload.chunk_overlap)
    if not chunks:
        raise HTTPException(status_code=400, detail="Texto vacío o inválido")
    count = _safe_add_chunks(
        session_id,
        chunks,
        title=payload.title,
        author=payload.author,
        source_type="text",
        source_ref="texto plano",
    )
    return {
        "status": "success",
        "chunks_created": count,
        "metadata": {"title": payload.title, "author": payload.author},
    }


@app.post("/api/upload-url")
def upload_url(payload: URLUploadRequest, request: Request, response: Response):
    session_id = get_session_id(request, response)
    try:
        title, text = ingestion.extract_url(payload.url)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"No se pudo obtener la URL: {exc}")

    chunks = ingestion.chunk_text(text, payload.chunk_size, payload.chunk_overlap)
    if not chunks:
        raise HTTPException(status_code=400, detail="La página no contiene texto utilizable")
    count = _safe_add_chunks(
        session_id,
        chunks,
        title=title,
        author="Autor desconocido",
        source_type="url",
        source_ref=payload.url,
    )
    return {
        "status": "success",
        "chunks_created": count,
        "metadata": {"title": title, "url": payload.url},
    }


@app.post("/api/upload-pdf")
async def upload_pdf(
    request: Request,
    response: Response,
    file: UploadFile = File(...),
    title: Optional[str] = None,
    author: Optional[str] = None,
):
    session_id = get_session_id(request, response)
    contents = await file.read()
    pages = ingestion.extract_pdf_pages(contents)
    if not pages:
        raise HTTPException(status_code=400, detail="No se pudo extraer texto del PDF")
    pairs = ingestion.chunk_pages(pages, 1000, 150)
    if not pairs:
        raise HTTPException(status_code=400, detail="No se pudo extraer texto del PDF")
    chunks = [p[0] for p in pairs]
    chunk_pages = [p[1] for p in pairs]
    pdf_title = title or (file.filename or "PDF").replace(".pdf", "")
    count = _safe_add_chunks(
        session_id,
        chunks,
        title=pdf_title,
        author=(author or "").strip() or "Autor desconocido",
        source_type="pdf",
        source_ref=file.filename or "archivo.pdf",
        pages=chunk_pages,
    )
    return {
        "status": "success",
        "chunks_created": count,
        "metadata": {"title": pdf_title, "filename": file.filename, "pages": len(pages)},
    }


@app.delete("/api/clear-rag")
def clear_rag(request: Request, response: Response):
    session_id = get_session_id(request, response)
    rag.clear(session_id)
    return {"status": "success"}


@app.get("/api/rag-status")
def rag_status(request: Request, response: Response):
    session_id = get_session_id(request, response)
    return rag.status(session_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
