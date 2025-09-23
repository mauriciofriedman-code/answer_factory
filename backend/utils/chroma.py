import chromadb
from chromadb.config import Settings
import os
from typing import List, Dict

# Initialize Chroma client
client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./data/chroma"
))

# Get or create collection
collection = client.get_or_create_collection(name="educational_content")

def store_chunks_in_chroma(chunks: List[str], source: str = "unknown") -> None:
    """Store text chunks in Chroma database"""
    if not chunks:
        return
    
    # Generate IDs for chunks
    ids = [f"{source}_{i}" for i in range(len(chunks))]
    
    # Store metadata
    metadatas = [{"source": source, "chunk_index": i} for i in range(len(chunks))]
    
    # Add to collection
    collection.add(
        documents=chunks,
        metadatas=metadatas,
        ids=ids
    )

def get_chunks_from_chroma(query: str, n_results: int = 5) -> List[str]:
    """Retrieve relevant chunks from Chroma"""
    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results['documents'][0] if results['documents'] else []
    except Exception as e:
        print(f"Error retrieving from Chroma: {e}")
        return []

def clear_chroma_db() -> None:
    """Clear all data from Chroma database"""
    client.delete_collection(name="educational_content")
    global collection
    collection = client.create_collection(name="educational_content")