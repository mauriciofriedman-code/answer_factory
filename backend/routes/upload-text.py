from fastapi import APIRouter
from pydantic import BaseModel
from utils.chroma import store_chunks_in_chroma
from utils.processing import process_text

router = APIRouter()

class TextData(BaseModel):
    text: str
    chunk_size: int = 500
    chunk_overlap: int = 50

@router.post("/")
async def upload_text(data: TextData):
    try:
        # Process text into chunks
        chunks = process_text(data.text, data.chunk_size, data.chunk_overlap)
        
        # Store chunks in Chroma
        store_chunks_in_chroma(chunks, source="text_upload")
        
        return {
            "status": "success",
            "message": "Text processed and stored",
            "chunks_created": len(chunks)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}