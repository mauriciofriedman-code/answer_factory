"""Smoke test del laboratorio.

Uso:
  1. Asegúrate de tener OPENAI_API_KEY en backend/.env
  2. Arranca el backend:   uvicorn app:app --port 8000
  3. En otra terminal:     python smoke_test.py

Por defecto apunta a http://localhost:8000. Cambia BASE si lo deseas.
"""
from __future__ import annotations

import json
import sys
import time
from typing import Any

import requests

BASE = "http://localhost:8000"
TIMEOUT = 90  # las llamadas a OpenAI pueden tardar


# ──────────────────────────────────────────────────────────────────────
# Helpers de presentación
# ──────────────────────────────────────────────────────────────────────

def banner(title: str) -> None:
    print()
    print("═" * 72)
    print(f" {title}")
    print("═" * 72)


def show(label: str, payload: Any, max_chars: int = 600) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2) if not isinstance(payload, str) else payload
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n… (+{len(text) - max_chars} chars truncados)"
    print(f"\n▸ {label}\n{text}")


# ──────────────────────────────────────────────────────────────────────
# Sesión HTTP — preserva la cookie answer_factory_session entre requests
# ──────────────────────────────────────────────────────────────────────

session = requests.Session()


def call(method: str, path: str, **kwargs) -> dict:
    url = f"{BASE}{path}"
    started = time.perf_counter()
    resp = session.request(method, url, timeout=TIMEOUT, **kwargs)
    elapsed = (time.perf_counter() - started) * 1000
    print(f"[{method} {path}] → {resp.status_code} · {elapsed:.0f} ms")
    if resp.status_code >= 400:
        print(f"  body: {resp.text[:400]}")
        return {"_status": resp.status_code, "_error": resp.text}
    try:
        return resp.json()
    except ValueError:
        return {"_status": resp.status_code, "_text": resp.text}


# ──────────────────────────────────────────────────────────────────────
# Pruebas
# ──────────────────────────────────────────────────────────────────────

def test_health_and_config() -> None:
    banner("1 · Health & config público")
    show("/health", call("GET", "/health"))
    show("/api/config", call("GET", "/api/config"))


def test_generate_basic() -> None:
    banner("2 · /api/generate — prompt vago vs prompt CRAFT-D")

    vague = {
        "prompt": "Hazme una rúbrica.",
        "model": "gpt-4o-mini",
        "style": "natural",
        "temperature": 0.5,
        "top_p": 0.9,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "max_tokens": 250,
        "use_rag": False,
        "stop_sequences": [],
    }
    show("vago — request", vague)
    show("vago — respuesta", call("POST", "/api/generate", json=vague))

    crafted = {
        **vague,
        "style": "craftd",
        "prompt": (
            "## CONTEXTO\n5° de primaria, materia de Ciencias Naturales, sesión de 50 min.\n\n"
            "## ROL\nDiseñador instruccional con experiencia en evaluación auténtica.\n\n"
            "## AUDIENCIA\nDocente de grupo, sin formación específica en evaluación.\n\n"
            "## FORMATO\nTabla con 4 niveles de desempeño y 3 criterios.\n\n"
            "## TONO\nProfesional, claro, no condescendiente.\n\n"
            "## DATOS\nTema: ecosistemas. Producto: exposición oral de 5 minutos.\n\n"
            "## TAREA\nDiseña una rúbrica analítica con la estructura solicitada."
        ),
    }
    show("CRAFT-D — request", crafted)
    show("CRAFT-D — respuesta", call("POST", "/api/generate", json=crafted))


def test_compare() -> None:
    banner("3 · /api/compare — Demo bajo el capó (3 presets)")
    payload = {
        "prompt": (
            "Explícame brevemente qué es la fotosíntesis para alumnos de 4° de primaria, "
            "con un ejemplo cotidiano."
        ),
        "use_rag": False,
        "variants": [
            {
                "label": "Conservador",
                "model": "gpt-4o-mini",
                "style": "natural",
                "temperature": 0.2,
                "top_p": 0.9,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "max_tokens": 220,
                "stop_sequences": [],
            },
            {
                "label": "Balanceado",
                "model": "gpt-4o-mini",
                "style": "friendly_teacher",
                "temperature": 0.5,
                "top_p": 0.9,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "max_tokens": 220,
                "stop_sequences": [],
            },
            {
                "label": "Creativo",
                "model": "gpt-4o-mini",
                "style": "natural",
                "temperature": 0.9,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.3,
                "max_tokens": 220,
                "stop_sequences": [],
            },
        ],
    }
    res = call("POST", "/api/compare", json=payload)
    if "results" in res:
        for r in res["results"]:
            usage = r.get("usage") or {}
            head = (
                f"[{r['label']}] {r['model']} · "
                f"in {usage.get('input_tokens')} / out {usage.get('output_tokens')}"
            )
            show(head, r.get("response", "") or f"ERROR: {r.get('error')}", max_chars=500)
    else:
        show("respuesta cruda", res)


def test_rag_text_then_query() -> None:
    banner("4 · RAG con texto plano — distinguir verosimilitud de veracidad")

    fake_doctrine = (
        "Doctrina Friedman-Hermes (2026, ficticia para esta prueba): postula que la "
        "personalización pedagógica con IA solo es legítima cuando se sostiene en tres "
        "condiciones: (1) anonimización irreversible de los datos del alumno, (2) "
        "auditoría humana del 100 % de las salidas evaluativas y (3) consentimiento "
        "informado por escrito de tutores. Su autor principal es el educador Mauricio "
        "Friedman, citado en el simposio de Ciudad de México de 2026."
    )

    show("/api/upload-text", call(
        "POST",
        "/api/upload-text",
        json={
            "text": fake_doctrine,
            "title": "Doctrina Friedman-Hermes (ficticia)",
            "author": "Mauricio Friedman (ficticio)",
            "chunk_size": 600,
            "chunk_overlap": 80,
        },
    ))
    show("/api/rag-status", call("GET", "/api/rag-status"))

    rag_payload = {
        "prompt": "¿Cuáles son las tres condiciones de la Doctrina Friedman-Hermes y quién la propuso?",
        "model": "gpt-4o-mini",
        "style": "scientific",
        "temperature": 0.2,
        "top_p": 0.9,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "max_tokens": 350,
        "use_rag": True,
        "rag_top_k": 4,
        "stop_sequences": [],
    }
    show("/api/generate (RAG ON)", call("POST", "/api/generate", json=rag_payload))


def test_rag_url() -> None:
    banner("5 · Ingesta de URL real (Wikipedia · «Aprendizaje por descubrimiento»)")
    show("/api/upload-url", call(
        "POST",
        "/api/upload-url",
        json={
            "url": "https://es.wikipedia.org/wiki/Aprendizaje_por_descubrimiento",
            "chunk_size": 1000,
            "chunk_overlap": 150,
        },
    ))
    show("/api/rag-status", call("GET", "/api/rag-status"))

    show("/api/generate (RAG ON con URL)", call(
        "POST",
        "/api/generate",
        json={
            "prompt": "Según la fuente subida, ¿qué autor es central en el aprendizaje por descubrimiento y cuáles son sus ideas principales?",
            "model": "gpt-4o-mini",
            "style": "friendly_teacher",
            "temperature": 0.3,
            "top_p": 0.9,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "max_tokens": 350,
            "use_rag": True,
            "rag_top_k": 4,
            "stop_sequences": [],
        },
    ))


def test_session_isolation() -> None:
    banner("6 · Aislamiento por sesión")
    print("Pista: una nueva session de requests inicia su propia cookie y debería ver 0 fragmentos.")
    fresh = requests.Session()
    resp = fresh.get(f"{BASE}/api/rag-status", timeout=TIMEOUT)
    print(f"[GET /api/rag-status (sesión nueva)] → {resp.status_code}")
    show("status sesión nueva", resp.json())


def test_clear_rag() -> None:
    banner("7 · Limpiar fuentes de mi sesión")
    show("/api/clear-rag", call("DELETE", "/api/clear-rag"))
    show("/api/rag-status", call("GET", "/api/rag-status"))


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main() -> int:
    try:
        ping = requests.get(f"{BASE}/health", timeout=5)
        ping.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        print(f"❌ No puedo conectarme a {BASE}: {exc}")
        print("   Arranca el backend primero:  uvicorn app:app --port 8000")
        return 1

    health = ping.json()
    if not health.get("openai_configured"):
        print("⚠ OPENAI_API_KEY no está configurada — las pruebas de generación devolverán texto de prueba.")
    else:
        print("✓ Backend vivo y OPENAI_API_KEY presente.")

    test_health_and_config()
    test_generate_basic()
    test_compare()
    test_rag_text_then_query()
    test_rag_url()
    test_session_isolation()
    test_clear_rag()

    banner("FIN")
    print("Si llegaste hasta aquí sin errores rojos arriba, el laboratorio responde como debe.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
