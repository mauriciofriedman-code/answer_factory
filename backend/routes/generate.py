from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

# Fixed imports - no "backend." prefix
from utils.model import generate_response_from_model
from utils.chroma import get_chunks_from_chroma

router = APIRouter()

class PromptData(BaseModel):
    prompt: str
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: Optional[int] = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    max_tokens: int = 500
    stop_sequences: Optional[list] = None
    style: str = "neutral"
    use_rag: bool = False

@router.post("/")
async def generate_response(data: PromptData):
    # Apply style to prompt
    styled_prompt = apply_style_to_prompt(data.prompt, data.style)
    
    # Get RAG chunks if enabled
    chunks = []
    if data.use_rag:
        chunks = get_chunks_from_chroma(data.prompt)
    
    # Generate response with model
    response = generate_response_from_model(
        prompt=styled_prompt,
        chunks=chunks,
        temperature=data.temperature,
        top_p=data.top_p,
        top_k=data.top_k,
        frequency_penalty=data.frequency_penalty,
        presence_penalty=data.presence_penalty,
        max_tokens=data.max_tokens,
        stop_sequences=data.stop_sequences
    )
    
    return {"response": response, "chunks_used": len(chunks)}

def apply_style_to_prompt(prompt: str, style: str) -> str:
    """Apply response style to prompt"""
    styles = {
        "scientist": f"You are a scientist answering with academic rigor. {prompt}",
        "teacher": f"You are a loving teacher who answers with care and understanding. {prompt}",
        "friendly": f"You are a friendly assistant. {prompt}",
        "neutral": prompt
    }
    return styles.get(style, prompt)