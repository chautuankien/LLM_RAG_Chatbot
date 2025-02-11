from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document

from uuid import uuid4
from collections.abc import Sequence
import streamlit as st
from dotenv import load_dotenv
import os
load_dotenv()

MONGODB_ATLAS_CLUSTER_URI = os.getenv("MONGODB_ATLAS_CLUSTER_URI")

def add_files_to_mongodb(documents: Sequence[list[Document]], vectorstore: VectorStore) -> None:
    # Add documents into MongoDB   
    for i in range(len(documents)):
        if documents[i] is not None:
            vectorstore.add_documents(documents=documents[i], ids=[str(uuid4()) for _ in range(len(documents[i]))])

def mongodb_initialization(URI_link: str, collection_name: str,  embedding_model: Embeddings) -> VectorStore:
    """
    Connect to MongoDB database
    Args:
        URI_link (str): URI link to MongoDB database
        collection_name (str): Collection name in MongoDB database
    Returns:
        MongoDB: MongoDB database
    """    
    try:
        # initialize MongoDB python client
        client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)

        DB_NAME = "llm_rag_db"
        COLLECTION_NAME = collection_name
        ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"

        MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]

        vectorstore = MongoDBAtlasVectorSearch(
            collection=MONGODB_COLLECTION,
            embedding=embedding_model,
            index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
            relevance_score_fn="cosine",
        )
        vectorstore.create_vector_search_index(dimensions=1536)
        return vectorstore      
    except Exception as e:
        st.error(body=f"Error connecting to MongoDB: {e}. Creating local MongoDB database.")
        
        return None