import openai

def create_embeddings(text: str) -> list:
    """
    Genera los embeddings para un texto usando OpenAI.
    
    :param text: El texto para generar el embedding.
    :return: Lista de embeddings.
    """
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']
