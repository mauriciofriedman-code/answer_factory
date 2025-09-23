import openai
from dotenv import load_dotenv
import os

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_response(prompt: str, temperature: float, top_p: float, max_tokens: int, penalty_frequency: float, penalty_presence: float, top_k: int):
    """Genera una respuesta con OpenAI basado en el prompt y parámetros."""
    response = openai.Completion.create(
        engine="text-davinci-003",  # Modelo de OpenAI
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        n=top_k,
        stop=None
    )
    return response.choices[0].text.strip()
