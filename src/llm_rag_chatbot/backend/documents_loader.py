import os
import re
from langchain_community.document_loaders import RecursiveUrlLoader, PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from bs4 import BeautifulSoup

def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()

def crawl_web(url_data) -> list[Document]:
    """
    Crawl data from web
    """
    loader = RecursiveUrlLoader(url=url_data,
                                 extractor=bs4_extractor,
                                 max_depth=4)
    docs = loader.load()
    print(f"Loaded {len(docs)} documents")
    
    # Create text splitter and split documents into docs
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    all_splits = text_splitter.split_documents(docs)
    print(f"Split into {len(all_splits)} splits")

    return all_splits

def load_pdf_from_local(directory: str) -> list[Document]:
    """
    Load all PDF files from local directory
    Args:
        directory (str): Directory path (e.g. "data") # where the PDF files are stored
    Returns:
        documents (list): List of documents
    """
    for root, dirs, files in os.walk(directory):                                            
        for file in files:
            if file.endswith(".pdf"):
                print(file)
                loader = PDFMinerLoader(os.path.join(root, file))

    docs = loader.load()
    print(f"Loaded {len(docs)} documents")

    # Create text splitter and split documents into docs
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    all_splits = text_splitter.split_documents(docs)
    print(f"Split into {len(all_splits)} splits")

    return all_splits       

