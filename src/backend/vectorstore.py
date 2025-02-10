from langchain_openai import OpenAIEmbeddings
from langchain_milvus import Milvus
from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.documents import Document

from backend.documents_loader import crawl_web, load_pdf_from_local

from uuid import uuid4
from collections.abc import Sequence

import streamlit as st

def add_files_to_milvus(documents: Sequence[list[Document]], vectorstore: VectorStore) -> None:
    # Add documents into Milvus
    for i in range(len(documents)):
        vectorstore.add_documents(documents=documents[i])
    print(f'vectorstore: {vectorstore}')


def add_data_to_milvus_url(url: str, URI_link: str, collection_name: str, doc_name: str, embedding_choice: str) -> Milvus:
    """"
    Crawling data from URL and store in Milvus database
    Args:
        url (str): URL to crawl data
        URI_link (str): URI link to Milvus database
        collection_name (str): Collection name in Milvus database
        embedding_choice (str): Embedding model choice
    Returns:
        Milvus: Milvus database
    """
    documents = crawl_web(url_data=url)

    # Create Embedding model
    print(embedding_choice)
    if embedding_choice == "OpenAI":
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    # Update metadata for each document
    for doc in documents:
        metadata = {
            'source': doc.metadata.get('source') or '',
            'content_type': doc.metadata.get('content_type') or 'text/plain',
            'title': doc.metadata.get('title') or '',
            'desciption': doc.metadata.get('desciption') or '',
            'language': doc.metadata.get('language') or 'en',
            'doc_name': doc_name,
            'start_index': doc.metadata.get('start_index') or 0,
        }
        doc.metadata = metadata

    uuids = [str(uuid4()) for _ in range(len(documents))]

    vectorstore = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": URI_link},
        collection_name=collection_name,
        drop_old=True
    )
    vectorstore.add_documents(documents=documents, ids=uuids)
    print(f'vectorstore: {vectorstore}')

    return vectorstore
   

def milvus_initialization(URI_link: str, collection_name: str,  embedding_model: Embeddings) -> VectorStore:
    """
    Connect to Milvus database
    Args:
        URI_link (str): URI link to Milvus database
        collection_name (str): Collection name in Milvus database
    Returns:
        Milvus: Milvus database
    """    
    try:
        vectorstore = Milvus(
            embedding_function=embedding_model,
            connection_args={"uri": URI_link},
            collection_name=collection_name,
        )
        return vectorstore
    except Exception as e:
        st.error(body=f"Error connecting to Milvus: {e}. Creating local Milvus database.")
        
        local_link: str = "./milvus_example.db"
        vectorstore = Milvus(
            embedding_function=embedding_model,
            connection_args={"uri": local_link},
            collection_name=collection_name,
        )
        return vectorstore
