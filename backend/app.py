from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import openai
import os
from dotenv import load_dotenv
import json
from datetime import datetime

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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# In-memory storage for RAG content with metadata
rag_storage = {
    "chunks": [],
    "metadata": []
}

# Pydantic models
class GenerateRequest(BaseModel):
    prompt: str
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: Optional[int] = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    max_tokens: int = 500
    style: str = "natural"
    use_rag: bool = False
    stop_sequences: List[str] = []

class TextUploadRequest(BaseModel):
    text: str
    chunk_size: int = 500
    chunk_overlap: int = 50
    # Metadata fields for plain text
    title: Optional[str] = "Documento sin título"
    author: Optional[str] = "Autor desconocido"
    source: Optional[str] = "Texto plano"

class URLUploadRequest(BaseModel):
    url: str
    # These will be auto-extracted when possible
    title: Optional[str] = None
    author: Optional[str] = None

@app.get("/")
def root():
    return {"message": "The Answer Factory API is running! 🏭🤖"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "message": "API funcionando correctamente",
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "rag_chunks": len(rag_storage["chunks"])
    }

@app.post("/api/generate")
async def generate_response(request: GenerateRequest):
    """Generate AI response with customizable parameters and styles"""
    try:
        # Style prompts for different personas
        style_prompts = {
            "natural": None,  # No system message - pure model response
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
        
        # Get the style prompt
        system_message = style_prompts.get(request.style, None)
        
        # If OpenAI key is not configured, return a test response
        if not openai.api_key:
            return {
                "response": f"[Modo de Prueba - OpenAI no configurado]\n\nEstilo: {request.style}\nPregunta: {request.prompt}\n\n(Configure OPENAI_API_KEY para respuestas reales)",
                "chunks_used": 0,
                "sources": []
            }
        
        # Prepare context if RAG is enabled
        context_text = ""
        sources_used = []
        
        if request.use_rag and rag_storage["chunks"]:
            # Simple relevance: use all chunks for now
            # In production, you'd use embeddings and similarity search
            context_chunks = rag_storage["chunks"][:5]  # Use top 5 chunks
            context_metadata = rag_storage["metadata"][:5]
            
            context_text = "\n\n".join([
                f"[Fuente: {meta.get('title', 'Sin título')} - {meta.get('author', 'Autor desconocido')} - Página {meta.get('page', 'N/A')}]\n{chunk}"
                for chunk, meta in zip(context_chunks, context_metadata)
            ])
            
            sources_used = [
                {
                    "title": meta.get("title", "Sin título"),
                    "author": meta.get("author", "Autor desconocido"),
                    "page": meta.get("page", "N/A"),
                    "type": meta.get("type", "unknown")
                }
                for meta in context_metadata
            ]
            
            # Add context to the prompt
            augmented_prompt = f"""Contexto proporcionado:
{context_text}

Pregunta del usuario: {request.prompt}

Por favor, responde basándote en el contexto proporcionado y cita las fuentes cuando sea relevante."""
        else:
            augmented_prompt = request.prompt
        
        # Build messages for OpenAI
        messages = []
        if system_message:  # Only add system message if style is not "natural"
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": augmented_prompt})
        
        # Create the OpenAI chat completion
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=request.temperature,
            top_p=request.top_p,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            max_tokens=request.max_tokens,
            stop=request.stop_sequences if request.stop_sequences else None
        )
        
        # Extract the response text
        ai_response = response.choices[0].message.content
        
        return {
            "response": ai_response,
            "chunks_used": len(sources_used),
            "sources": sources_used
        }
    
    except Exception as e:
        print(f"Error in generate_response: {str(e)}")
        return {
            "response": f"Error al generar respuesta: {str(e)}",
            "chunks_used": 0,
            "sources": []
        }

@app.post("/api/upload-text")
async def upload_text(request: TextUploadRequest):
    """Upload plain text for RAG embeddings with metadata"""
    try:
        # Split text into chunks
        text_length = len(request.text)
        chunk_size = request.chunk_size
        chunk_overlap = request.chunk_overlap
        
        chunks = []
        for i in range(0, text_length, chunk_size - chunk_overlap):
            chunk = request.text[i:i + chunk_size]
            chunks.append(chunk)
            
            # Add metadata for each chunk
            rag_storage["chunks"].append(chunk)
            rag_storage["metadata"].append({
                "type": "text",
                "title": request.title,
                "author": request.author,
                "source": request.source,
                "page": f"{i // chunk_size + 1}",  # Approximate page number
                "upload_time": datetime.now().isoformat()
            })
        
        return {
            "status": "success",
            "message": f"Texto '{request.title}' procesado correctamente",
            "chunks_created": len(chunks),
            "metadata": {
                "title": request.title,
                "author": request.author
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload-url")
async def upload_url(request: URLUploadRequest):
    """Extract and store content from URL with metadata"""
    try:
        # TODO: Implement actual URL fetching with beautifulsoup4
        # For now, simulate with metadata
        
        # Simulated extraction (in production, use BeautifulSoup)
        simulated_title = request.title or f"Contenido de {request.url}"
        simulated_author = request.author or "Extraído de web"
        simulated_content = f"[Contenido simulado de {request.url}]"
        
        # Create chunks
        chunks = [simulated_content]  # In production, properly chunk the content
        
        for i, chunk in enumerate(chunks):
            rag_storage["chunks"].append(chunk)
            rag_storage["metadata"].append({
                "type": "url",
                "title": simulated_title,
                "author": simulated_author,
                "source": request.url,
                "page": "Web",
                "upload_time": datetime.now().isoformat()
            })
        
        return {
            "status": "success",
            "message": f"URL procesada: {request.url}",
            "chunks_created": len(chunks),
            "metadata": {
                "title": simulated_title,
                "author": simulated_author,
                "url": request.url
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Process PDF for RAG with metadata extraction"""
    try:
        # TODO: Implement actual PDF processing with pdfminer.six
        # For now, simulate metadata extraction
        
        contents = await file.read()
        file_size = len(contents)
        
        # Simulated metadata (in production, extract from PDF)
        pdf_title = file.filename.replace('.pdf', '')
        pdf_author = "Autor del PDF"  # Extract from PDF metadata
        
        # Simulate chunks with page numbers
        estimated_chunks = max(1, file_size // 5000)
        
        for i in range(estimated_chunks):
            rag_storage["chunks"].append(f"[Contenido del PDF {file.filename} - Chunk {i+1}]")
            rag_storage["metadata"].append({
                "type": "pdf",
                "title": pdf_title,
                "author": pdf_author,
                "source": file.filename,
                "page": str(i + 1),  # Actual page number in production
                "upload_time": datetime.now().isoformat()
            })
        
        return {
            "status": "success",
            "message": f"PDF procesado: {file.filename}",
            "chunks_created": estimated_chunks,
            "metadata": {
                "title": pdf_title,
                "author": pdf_author,
                "pages": estimated_chunks
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/clear-rag")
async def clear_rag():
    """Clear all RAG storage"""
    rag_storage["chunks"] = []
    rag_storage["metadata"] = []
    return {"status": "success", "message": "RAG storage cleared"}

@app.get("/api/rag-status")
async def rag_status():
    """Get current RAG storage status"""
    return {
        "total_chunks": len(rag_storage["chunks"]),
        "sources": [
            {
                "title": meta.get("title"),
                "author": meta.get("author"),
                "type": meta.get("type")
            }
            for meta in rag_storage["metadata"]
        ][:10]  # Show first 10 sources
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)