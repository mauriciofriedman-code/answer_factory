"""Smoke focalizado: 3 providers + 3 estilos + chequeo anti-Markdown."""
from __future__ import annotations

import io
import re
import sys
import time

import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE = "http://localhost:8000"
TIMEOUT = 120

PROMPT = (
    "Un docente de bachillerato me pregunta qué es el scaffolding pedagógico, "
    "de dónde viene la idea y cómo aplicarlo en una clase de literatura. "
    "Respóndele en tres párrafos."
)

CASES = [
    ("OpenAI · gpt-4o",          "gpt-4o",            "scientific"),
    ("Anthropic · sonnet-4-6",   "claude-sonnet-4-6", "friendly_teacher"),
    ("Google · gemini-2.5-pro",  "gemini-2.5-pro",    "natural"),
]

MARKDOWN_RX = re.compile(r"(\*\*|^#{1,6}\s|^\s*[-*]\s|^\s*\d+\.\s)", re.MULTILINE)


def main() -> int:
    s = requests.Session()
    health = s.get(f"{BASE}/health", timeout=10).json()
    print(f"health: {health['status']} · models: {len(health['models'])}\n")

    rows = []
    for label, model, style in CASES:
        payload = {
            "prompt": PROMPT,
            "model": model,
            "style": style,
            "temperature": 0.5,
            "top_p": 0.9,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "max_tokens": 1500,
            "use_rag": False,
            "stop_sequences": [],
        }
        t0 = time.perf_counter()
        try:
            r = s.post(f"{BASE}/api/generate", json=payload, timeout=TIMEOUT)
            dt = (time.perf_counter() - t0) * 1000
            data = r.json()
        except Exception as exc:
            print(f"FAIL {label} -- error de red: {exc}\n")
            rows.append((label, "NETWORK_ERROR", 0, 0, ""))
            continue

        if r.status_code >= 400:
            print(f"FAIL {label} -- HTTP {r.status_code}: {data}\n")
            rows.append((label, f"HTTP {r.status_code}", dt, 0, ""))
            continue

        text = data.get("response", "")
        usage = data.get("usage") or {}
        md_hits = MARKDOWN_RX.findall(text)
        verdict = "CLEAN" if not md_hits else f"MARKDOWN ({len(md_hits)} marcas)"

        print(f"=== {label} === [{model}/{style}] · {dt:.0f}ms · {verdict}")
        print(f"  in={usage.get('input_tokens')} out={usage.get('output_tokens')} finish={data.get('finish_reason')}")
        print(f"  texto (primeros 480 chars):")
        print("  " + text[:480].replace("\n", "\n  "))
        print()
        rows.append((label, verdict, dt, usage.get("output_tokens") or 0, text[:80]))

    print("─" * 72)
    print(f"{'CASO':<32} {'FORMA':<22} {'ms':>7}  {'tok':>5}")
    print("─" * 72)
    for label, verdict, dt, tok, _ in rows:
        print(f"{label:<32} {verdict:<22} {dt:>7.0f}  {tok:>5}")
    fails = [r for r in rows if r[1] != "CLEAN"]
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
