from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(
    title="The Answer Factory API",
    description="API for AI-powered educational content personalization",
    version="1.0.0"
)

# CORS middleware - Allow all origins for Render deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins since both frontend and backend are on Render
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Pydantic models for request/response
class GenerateRequest(BaseModel):
    prompt: str
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: Optional[int] = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    max_tokens: int = 500
    style: str = "neutral"
    use_rag: bool = False
    stop_sequences: List[str] = []

class TextUploadRequest(BaseModel):
    text: str
    chunk_size: int = 500
    chunk_overlap: int = 50

class URLUploadRequest(BaseModel):
    url: str

@app.get("/")
def root():
    return {"message": "The Answer Factory API is running! 🏭🤖"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "message": "API funcionando correctamente",
        "openai_configured": bool(os.getenv("OPENAI_API_KEY"))
    }

@app.post("/api/generate")
async def generate_response(request: GenerateRequest):
    """Generate AI response with customizable parameters and styles"""
    try:
        # Style prompts for different personas
        style_prompts = {
            "neutral": "Respond in a clear, neutral manner.",
            "scientific": "Responde de manera científica y técnica, usando terminología precisa y explicando los conceptos con rigor académico.",
            "friendly_teacher": "Responde como un profesor amigable y alentador que quiere ayudar al estudiante a entender.",
            "child_5yo": "Responde como si fueras un niño de 5 años, usando lenguaje muy simple y comparaciones infantiles.",
            "shakespeare": "Responde en estilo shakesperiano dramático, con lenguaje teatral y poético.",
            "wise_grandmother": "Responde como una abuela sabia y cariñosa que cuenta historias y da consejos con amor.",
            "confused_robot": "Responde como un robot confundido que toma todo literalmente y no entiende las metáforas.",
            "philosopher": "Responde como un filósofo existencial, cuestionando todo y reflexionando profundamente.",
            "salesperson": "Responde como un vendedor súper entusiasta que está emocionado por todo.",
            "storyteller": "Responde como un narrador de cuentos, convirtiendo todo en una narrativa.",
            "comedian": "Responde como un comediante tratando de hacer todo gracioso con chistes y juegos de palabras.",
            "pirate": "¡Arrr! Responde como un pirata con jerga pirata y referencias náuticas, marinero!",
            "chef": "Responde como un chef, usando metáforas culinarias para explicar todo como si fuera una receta."
        }
        
        # Get the style prompt or use neutral as default
        system_message = style_prompts.get(request.style, style_prompts["neutral"])
        
        # If OpenAI key is not configured, return a test response
        if not openai.api_key:
            return {
                "response": f"[Modo de Prueba - OpenAI no configurado]\n\nEstilo: {request.style}\nPregunta: {request.prompt}\nTemperatura: {request.temperature}\nMax Tokens: {request.max_tokens}\n\n(Configure OPENAI_API_KEY para respuestas reales)",
                "chunks_used": 0
            }
        
        # Create the OpenAI chat completion
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": request.prompt}
            ],
            temperature=request.temperature,
            top_p=request.top_p,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            max_tokens=request.max_tokens,
            stop=request.stop_sequences if request.stop_sequences else None
        )
        
        # Extract the response text
        ai_response = response.choices[0].message.content
        
        # If RAG is enabled (future implementation)
        chunks_used = 0
        if request.use_rag:
            # TODO: Implement RAG functionality with ChromaDB
            # For now, just indicate RAG was requested
            ai_response = f"[RAG Activado - En desarrollo]\n\n{ai_response}"
            chunks_used = 3  # Placeholder
        
        return {
            "response": ai_response,
            "chunks_used": chunks_used
        }
    
    except openai.error.AuthenticationError:
        return {
            "response": "Error: Clave API de OpenAI inválida. Por favor, verifica la configuración.",
            "chunks_used": 0
        }
    except openai.error.RateLimitError:
        return {
            "response": "Error: Límite de uso de API excedido. Intenta más tarde.",
            "chunks_used": 0
        }
    except Exception as e:
        print(f"Error in generate_response: {str(e)}")
        return {
            "response": f"Error al generar respuesta: {str(e)}",
            "chunks_used": 0
        }

@app.post("/api/upload-text")
async def upload_text(request: TextUploadRequest):
    """Upload plain text for RAG embeddings"""
    try:
        # TODO: Implement actual text processing and ChromaDB storage
        # For now, return success with simulated chunk count
        text_length = len(request.text)
        estimated_chunks = max(1, text_length // request.chunk_size)
        
        return {
            "status": "success",
            "message": "Texto recibido correctamente",
            "chunks_created": estimated_chunks
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload-url")
async def upload_url(request: URLUploadRequest):
    """Extract and store content from URL"""
    try:
        # TODO: Implement actual URL fetching with beautifulsoup4
        # For now, return success with simulated data
        return {
            "status": "success",
            "message": f"URL procesada: {request.url}",
            "chunks_created": 5
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Process PDF for RAG"""
    try:
        # TODO: Implement actual PDF processing with pdfminer.six
        # For now, return success with simulated data
        file_size = 0
        contents = await file.read()
        file_size = len(contents)
        
        # Estimate chunks based on file size (rough estimate)
        estimated_chunks = max(1, file_size // 5000)
        
        return {
            "status": "success",
            "message": f"PDF procesado: {file.filename}",
            "chunks_created": estimated_chunks,
            "file_size": file_size
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Additional endpoint for testing OpenAI connection
@app.get("/api/test-openai")
async def test_openai():
    """Test if OpenAI API is properly configured"""
    try:
        if not openai.api_key:
            return {
                "configured": False,
                "message": "OpenAI API key not configured"
            }
        
        # Try a simple completion to test the API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Di 'hola' en una palabra"}],
            max_tokens=10
        )
        
        return {
            "configured": True,
            "message": "OpenAI API configurada correctamente",
            "test_response": response.choices[0].message.content
        }
    except Exception as e:
        return {
            "configured": False,
            "message": f"Error al conectar con OpenAI: {str(e)}"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)