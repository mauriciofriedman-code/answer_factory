# backend/config.py

import os

# Clave de API (puede estar en un archivo .env o ser proporcionada de forma segura)
API_KEY = os.getenv("API_KEY", "tu_clave_api_aqui")

# Ruta de Chroma
CHROMA_DB_PATH = "data/chroma"
