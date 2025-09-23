import openai
from typing import List, Optional
import os
from config import API_KEY

# Set OpenAI API key
openai.api_key = API_KEY

def generate_response_from_model(
    prompt: str,
    chunks: List[str] = [],
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: Optional[int] = None,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    max_tokens: int = 500,
    stop_sequences: Optional[List[str]] = None
) -> str:
    """Generate response using OpenAI API with RAG context if provided"""
    
    # Build context from chunks
    context = ""
    if chunks:
        context = "\n\nRelevant Context:\n" + "\n".join(chunks[:3])  # Use top 3 chunks
        prompt = prompt + context
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or "gpt-4" if you have access
            messages=[
                {"role": "system", "content": "You are an educational AI assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            max_tokens=max_tokens,
            stop=stop_sequences
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        return f"Error generating response: {str(e)}"