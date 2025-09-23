# backend/routes/upload_pdf.py

from fastapi import APIRouter, File, UploadFile
from backend.utils.chroma import store_chunks_in_chroma
from backend.utils.processing import process_pdf

router = APIRouter()

@router.post("/")
async def upload_pdf(file: UploadFile = File(...)):
    # Procesamos el PDF para extraer los chunks
    chunks = process_pdf(file)
    
    # Almacenamos los chunks en Chroma
    store_chunks_in_chroma(chunks)
    
    return {"response": "PDF procesado y almacenado correctamente"}
