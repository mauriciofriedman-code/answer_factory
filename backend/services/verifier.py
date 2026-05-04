"""Detector de alucinación: contrasta cada afirmación contra el RAG del docente.

Pipeline:
  1. Pedimos al modelo que extraiga las afirmaciones factuales de la respuesta.
  2. Para cada afirmación, recuperamos los chunks más cercanos del RAG.
  3. Le pedimos al modelo que clasifique: SUSTENTADA / CONTRADICHA / NO_HAY_EVIDENCIA.

No es un sistema de fact-checking absoluto: es una herramienta pedagógica para
mostrar la diferencia entre "lo que la IA afirma" y "lo que tu corpus respalda".
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from services import llm, rag


def _build_extraction_prompt(response_text: str) -> str:
    return (
        "Eres un asistente de fact-checking. Extrae de la siguiente respuesta de IA "
        "todas las afirmaciones factuales relevantes (hechos, citas, atribuciones, "
        "datos numéricos). Ignora opiniones, recomendaciones y frases de cortesía.\n\n"
        "Devuelve EXCLUSIVAMENTE un JSON válido con esta forma:\n"
        '{"claims": ["afirmación 1", "afirmación 2", "..."]}\n\n'
        "Máximo 8 afirmaciones, cada una en una sola oración.\n\n"
        "Respuesta a analizar:\n---\n" + response_text + "\n---"
    )


def _build_verdict_prompt(claim: str, context: str) -> str:
    return (
        "Eres un evaluador estricto. Analiza si la AFIRMACIÓN está respaldada, "
        "contradicha o sin evidencia respecto al CONTEXTO proporcionado.\n\n"
        f"AFIRMACIÓN: {claim}\n\n"
        f"CONTEXTO (fragmentos del corpus del docente):\n{context}\n\n"
        "Devuelve EXCLUSIVAMENTE un JSON válido con esta forma:\n"
        '{"verdict": "SUSTENTADA" | "CONTRADICHA" | "NO_HAY_EVIDENCIA", '
        '"explanation": "1-2 oraciones citando el fragmento si aplica"}\n'
        "Sé exigente: si el contexto no menciona explícitamente la afirmación, "
        'usa "NO_HAY_EVIDENCIA". Solo "SUSTENTADA" cuando el contexto la diga sin ambigüedad.'
    )


def _parse_json_loose(text: str) -> Dict[str, Any]:
    """Extrae JSON aún si viene rodeado de prosa o fences."""
    if not text:
        return {}
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        first_nl = cleaned.find("\n")
        if first_nl != -1:
            cleaned = cleaned[first_nl + 1 :]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start : end + 1]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}


def extract_claims(response_text: str, *, model: str) -> List[str]:
    if not response_text.strip():
        return []
    out = llm.generate(
        model=model,
        prompt=_build_extraction_prompt(response_text),
        style="natural",
        temperature=0.0,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        max_tokens=600,
        stop_sequences=None,
        context_blocks=[],
    )
    parsed = _parse_json_loose(out["text"])
    claims = parsed.get("claims") or []
    return [c for c in claims if isinstance(c, str) and c.strip()][:8]


def judge_claim(
    claim: str,
    *,
    session_id: str,
    model: str,
    n_chunks: int = 3,
) -> Dict[str, Any]:
    retrieved = rag.query(session_id, claim, n_results=n_chunks)
    chunks = retrieved.get("chunks") or []
    metas = retrieved.get("metadatas") or []

    if not chunks:
        return {
            "claim": claim,
            "verdict": "NO_HAY_EVIDENCIA",
            "explanation": "No se encontraron fragmentos en tu corpus relacionados con esta afirmación.",
            "context_chunks": [],
        }

    context_lines = []
    for chunk, meta in zip(chunks, metas):
        title = meta.get("title", "Sin título")
        page = meta.get("page", "?")
        context_lines.append(f"[{title} · fragmento {page}]\n{chunk}")
    context_text = "\n\n---\n\n".join(context_lines)

    out = llm.generate(
        model=model,
        prompt=_build_verdict_prompt(claim, context_text),
        style="natural",
        temperature=0.0,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        max_tokens=400,
        stop_sequences=None,
        context_blocks=[],
    )
    parsed = _parse_json_loose(out["text"])
    verdict = parsed.get("verdict") or "NO_HAY_EVIDENCIA"
    if verdict not in {"SUSTENTADA", "CONTRADICHA", "NO_HAY_EVIDENCIA"}:
        verdict = "NO_HAY_EVIDENCIA"

    return {
        "claim": claim,
        "verdict": verdict,
        "explanation": parsed.get("explanation", "")[:600],
        "context_chunks": [
            {
                "title": meta.get("title"),
                "page": meta.get("page"),
                "snippet": chunk[:240],
            }
            for chunk, meta in zip(chunks, metas)
        ],
    }


def verify(response_text: str, *, session_id: str, model: str) -> Dict[str, Any]:
    claims = extract_claims(response_text, model=model)
    judgments = [judge_claim(c, session_id=session_id, model=model) for c in claims]
    summary = {
        "total": len(judgments),
        "sustentadas": sum(1 for j in judgments if j["verdict"] == "SUSTENTADA"),
        "contradichas": sum(1 for j in judgments if j["verdict"] == "CONTRADICHA"),
        "sin_evidencia": sum(1 for j in judgments if j["verdict"] == "NO_HAY_EVIDENCIA"),
    }
    return {"summary": summary, "claims": judgments}
