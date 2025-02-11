from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from unstructured.documents.elements import Element
import streamlit as st
import io

from backend.vectorstore.milvus_vectorstore import add_files_to_milvus, add_data_to_milvus_url
from backend.vectorstore.mongodb_vectorstore import add_files_to_mongodb
from backend.utils.data_parser import pdf_parser, extract_and_convert_all_data

def handle_upload_files(llm: BaseLanguageModel | None, vectorstore: VectorStore) -> None:
    """
    Handle when user choose to upload local file
    """

    uploaded_file: io.BytesIO | None = st.file_uploader(label="Upload Files", accept_multiple_files=False)
    if uploaded_file is not None:
        with st.spinner(text="Process data..."):
            raw_data: list[Element] | None = pdf_parser(input_data=uploaded_file)
            text_docs, image_docs, table_docs = extract_and_convert_all_data(elements=raw_data, decription_model=llm)

            try:
                add_files_to_mongodb(
                    # documents=[text_docs, image_docs, table_docs],
                    documents=[text_docs],
                    vectorstore=vectorstore
                )
                st.success(body="Data loaded successfully!")
            except Exception as e:
                st.error(body=f"Error loading data: {e}")

def handle_url_input(embedding: Embeddings):
    """
    Handle when user choose to input URL
    """
    collection_name = st.text_input(
        "Collection name in database",
        "data_test",
        help="Type collection name you want to use to store in database",
    )
    doc_name = st.text_input(
        "Document name",
        "stack_ai",
        help="Type document name you want to store in"
    )
    url = st.text_input("URL")

    if st.button("Load data from URL"):
        if not collection_name:
            st.error("Please enter a collection name.")
            return
        with st.spinner("Loading data..."):
            try:
                add_data_to_milvus_url(
                    url,
                    'http://localhost:19530',
                    collection_name,
                    doc_name,
                    embedding_choice
                )
                st.success("Data loaded successfully!")
            except Exception as e:
                st.error(f"Error loading data: {e}")

