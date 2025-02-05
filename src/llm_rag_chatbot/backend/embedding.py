from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings.embeddings import Embeddings

def get_embeddings_model(embedding_choice: str, embedding_api_key: str) -> Embeddings:
    """
    Get embedding model for a given choice.
    """
    if embedding_choice == "OpenAI":
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=embedding_api_key)
    
    return embeddings