from fastapi import APIRouter
from services.openai_service import generate_response_with_style
from fastapi.responses import JSONResponse

router = APIRouter()

@router.post("/generate")
async def generate_text(prompt: str, style: str = "Científico", temperature: float = 0.7, top_p: float = 0.9, max_tokens: int = 300):
    try:
        response = generate_response_with_style(prompt, style, temperature, top_p, max_tokens)
        return JSONResponse(content={"answer": response}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
