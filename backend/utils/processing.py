from typing import List
import requests
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text
from io import BytesIO

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    
    return chunks

def process_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Process plain text into chunks"""
    # Clean text
    text = text.strip()
    
    # Create chunks
    return chunk_text(text, chunk_size, overlap)

def process_url(url: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Extract and process content from URL"""
    try:
        # Fetch URL content
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract text
        text = soup.get_text(separator=' ', strip=True)
        
        # Create chunks
        return chunk_text(text, chunk_size, overlap)
    except Exception as e:
        raise Exception(f"Error processing URL: {str(e)}")

def process_pdf(file_bytes: bytes, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Extract and process text from PDF"""
    try:
        # Extract text from PDF
        text = extract_text(BytesIO(file_bytes))
        
        # Create chunks
        return chunk_text(text, chunk_size, overlap)
    except Exception as e:
        raise Exception(f"Error processing PDF: {str(e)}")