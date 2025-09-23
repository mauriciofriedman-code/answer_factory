from fastapi import APIRouter, File, UploadFile
from utils.chroma import store_chunks_in_chroma
from utils.processing import process_pdf

router = APIRouter()

@router.post("/")
async def upload_pdf(
    file: UploadFile = File(...),
    chunk_size: int = 500,
    chunk_overlap: int = 50
):
    try:
        # Read PDF content
        content = await file.read()
        
        # Process PDF into chunks
        chunks = process_pdf(content, chunk_size, chunk_overlap)
        
        # Store chunks in Chroma
        store_chunks_in_chroma(chunks, source=file.filename)
        
        return {
            "status": "success",
            "message": f"PDF '{file.filename}' processed and stored",
            "chunks_created": len(chunks)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}