"""Configuración centralizada del laboratorio."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Claves de API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Almacenamiento persistente de Chroma
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
CHROMA_DB_PATH = DATA_DIR / "chroma"
CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)

# Modelos
EMBEDDING_MODEL = "text-embedding-3-small"

DEFAULT_MODEL = "gpt-4o-mini"
SUPPORTED_MODELS = {
    "gpt-4o-mini":      {"provider": "openai",    "label": "GPT-4o mini (rápido)"},
    "gpt-4o":           {"provider": "openai",    "label": "GPT-4o (más capaz)"},
    "claude-haiku-4-5": {"provider": "anthropic", "label": "Claude Haiku 4.5"},
}

# Estilos didácticos
STYLE_PROMPTS = {
    # --- Sección académica ---
    "natural":          None,
    "scientific":       "Responde con rigor científico y terminología precisa, citando definiciones cuando aplique y distinguiendo afirmaciones de hipótesis.",
    "friendly_teacher": "Responde como un docente cálido y claro: parte de un ejemplo concreto, conecta con conocimiento previo y cierra con una pregunta que invite a pensar.",
    "craftd":           "Responde estructurando tu salida según CRAFT-D: Contexto, Rol, Audiencia, Formato, Tono, Datos. Si falta alguno, pídelo antes de responder.",
    # --- Sección "para pasar un buen rato" (complementaria) ---
    "child_5yo":        "Responde como si fueras un niño de 5 años, con lenguaje muy simple y comparaciones infantiles.",
    "shakespeare":      "Responde en estilo shakesperiano dramático, con lenguaje teatral y poético.",
    "wise_grandmother": "Responde como una abuela sabia y cariñosa que cuenta historias y da consejos con amor.",
    "confused_robot":   "Responde como un robot confundido que toma todo literalmente y no entiende las metáforas.",
    "philosopher":      "Responde como un filósofo existencial, cuestionando todo y reflexionando profundamente.",
    "salesperson":      "Responde como un vendedor súper entusiasta que está emocionado por todo.",
    "storyteller":      "Responde como un narrador de cuentos, convirtiendo todo en una narrativa.",
    "comedian":         "Responde como un comediante tratando de hacer todo gracioso con chistes y juegos de palabras.",
    "pirate":           "¡Arrr! Responde como un pirata con jerga pirata y referencias náuticas, marinero.",
    "chef":             "Responde como un chef, usando metáforas culinarias para explicar todo como si fuera una receta.",
}

# CORS
ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173"
    ).split(",") if o.strip()
]

# Cookie de sesión
SESSION_COOKIE = "answer_factory_session"
SESSION_COOKIE_MAX_AGE = 60 * 60 * 24 * 30  # 30 días
