"""Cliente unificado para OpenAI, Anthropic y Google Gemini con SDK moderno."""
from typing import List, Optional, Dict, Any
from openai import OpenAI
from anthropic import Anthropic
from google import genai
from google.genai import types as genai_types

from config import (
    OPENAI_API_KEY,
    ANTHROPIC_API_KEY,
    GOOGLE_API_KEY,
    SUPPORTED_MODELS,
    STYLE_PROMPTS,
    BASE_SYSTEM,
    DEFAULT_MODEL,
)

_openai_client: Optional[OpenAI] = None
_anthropic_client: Optional[Anthropic] = None
_google_client: Optional[genai.Client] = None


def get_openai() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY no configurada")
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


def get_anthropic() -> Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        if not ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY no configurada")
        _anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
    return _anthropic_client


def get_google() -> genai.Client:
    global _google_client
    if _google_client is None:
        if not GOOGLE_API_KEY:
            raise RuntimeError("GOOGLE_API_KEY no configurada")
        _google_client = genai.Client(api_key=GOOGLE_API_KEY)
    return _google_client


def resolve_style(style: str) -> Optional[str]:
    return STYLE_PROMPTS.get(style)


def build_system_message(style: str) -> str:
    """BASE_SYSTEM siempre, con el overlay del estilo si lo hay."""
    overlay = STYLE_PROMPTS.get(style)
    if overlay:
        return f"{BASE_SYSTEM}\n\n---\n\n{overlay}"
    return BASE_SYSTEM


def build_rag_prompt(user_prompt: str, context_blocks: List[str]) -> str:
    if not context_blocks:
        return user_prompt
    context = "\n\n".join(context_blocks)
    return (
        "Contexto proporcionado por el docente (úsalo como fuente principal y cita "
        "explícitamente cuando lo uses; si la pregunta no se responde con este contexto, "
        "dilo en lugar de inventar):\n\n"
        f"{context}\n\n"
        f"Pregunta: {user_prompt}"
    )


def generate(
    *,
    model: str,
    prompt: str,
    style: str,
    temperature: float,
    top_p: float,
    frequency_penalty: float,
    presence_penalty: float,
    max_tokens: int,
    stop_sequences: Optional[List[str]],
    context_blocks: List[str],
    return_logprobs: bool = False,
    top_logprobs: int = 3,
) -> Dict[str, Any]:
    """Genera respuesta del modelo seleccionado y devuelve texto + uso."""
    if model not in SUPPORTED_MODELS:
        model = DEFAULT_MODEL

    provider = SUPPORTED_MODELS[model]["provider"]
    system_message = build_system_message(style)
    user_content = build_rag_prompt(prompt, context_blocks)

    if provider == "openai":
        client = get_openai()
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_content})

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "max_tokens": max_tokens,
        }
        if stop_sequences:
            kwargs["stop"] = stop_sequences
        if return_logprobs:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = max(1, min(int(top_logprobs), 5))

        resp = client.chat.completions.create(**kwargs)

        logprobs_payload = None
        if return_logprobs and resp.choices and resp.choices[0].logprobs:
            content = resp.choices[0].logprobs.content or []
            logprobs_payload = []
            for item in content:
                top = []
                for alt in (item.top_logprobs or []):
                    top.append({"token": alt.token, "logprob": alt.logprob})
                logprobs_payload.append(
                    {
                        "token": item.token,
                        "logprob": item.logprob,
                        "top_alternatives": top,
                    }
                )

        return {
            "text": resp.choices[0].message.content or "",
            "model": model,
            "input_tokens": resp.usage.prompt_tokens if resp.usage else None,
            "output_tokens": resp.usage.completion_tokens if resp.usage else None,
            "finish_reason": resp.choices[0].finish_reason,
            "logprobs": logprobs_payload,
        }

    if provider == "anthropic":
        client = get_anthropic()
        anthropic_model = {
            "claude-haiku-4-5":  "claude-haiku-4-5-20251001",
        }.get(model, model)

        # Anthropic 4.x rechaza enviar `temperature` y `top_p` simultáneamente.
        # Priorizamos `temperature` por ser la perilla principal del laboratorio.
        kwargs = {
            "model": anthropic_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": user_content}],
        }
        if system_message:
            kwargs["system"] = system_message
        if stop_sequences:
            kwargs["stop_sequences"] = stop_sequences

        resp = client.messages.create(**kwargs)
        text_parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        return {
            "text": "".join(text_parts),
            "model": model,
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
            "finish_reason": resp.stop_reason,
            "logprobs": None,
        }

    if provider == "google":
        client = get_google()
        cfg_kwargs: Dict[str, Any] = {
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": max_tokens,
        }
        if system_message:
            cfg_kwargs["system_instruction"] = system_message
        if stop_sequences:
            cfg_kwargs["stop_sequences"] = stop_sequences

        resp = client.models.generate_content(
            model=model,
            contents=user_content,
            config=genai_types.GenerateContentConfig(**cfg_kwargs),
        )

        usage = getattr(resp, "usage_metadata", None)
        finish = None
        candidates = getattr(resp, "candidates", None) or []
        if candidates:
            fr = getattr(candidates[0], "finish_reason", None)
            finish = getattr(fr, "name", None) or (str(fr) if fr is not None else None)

        return {
            "text": resp.text or "",
            "model": model,
            "input_tokens": getattr(usage, "prompt_token_count", None) if usage else None,
            "output_tokens": getattr(usage, "candidates_token_count", None) if usage else None,
            "finish_reason": finish,
            "logprobs": None,
        }

    raise RuntimeError(f"Proveedor no soportado: {provider}")
