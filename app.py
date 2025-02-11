import sys
sys.path.append("./src")
import json
import logging.config
import logging.handlers
import pathlib

from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory
from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.vectorstores.base import VectorStore
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

from frontend.frontend import initialize_app, setup_api_keys, setup_chat_interface, setup_data_source, handle_user_input

from backend.get_model import get_embeddings_model, get_llm_model
from backend.vectorstore.milvus_vectorstore import milvus_initialization
from backend.vectorstore.mongodb_vectorstore import mongodb_initialization
from backend.agent import get_retriever, get_llm_and_agent

logger = logging.getLogger(name="my_app")

def setup_logging() -> None:
    # Create log folder if not exists
    log_path = pathlib.Path("./log")
    log_path.mkdir(parents=True, exist_ok=True)

    config_file = pathlib.Path("src/configs/log_configs.json")
    with open(file=config_file, mode='r') as f:
        config = json.load(f)
    
    logging.config.dictConfig(config=config)

def main() -> None:
    # Setup logging
    setup_logging()

    # Initialize frontend
    logger.info(msg="Starting frontend initialization")
    initialize_app()
    model_choice, embedding_choice, llm_api_key, embedding_api_key, collection_name = setup_api_keys()
    msgs: StreamlitChatMessageHistory = setup_chat_interface()
    logger.info(msg="Done frontend initialization")


    # Initialize backend
    logger.info(msg="Starting backend initialization")
    llm: BaseLanguageModel | None = get_llm_model(model_choice=model_choice, llm_api_key=llm_api_key)
    embedding: Embeddings | None = get_embeddings_model(embedding_choice=embedding_choice, embedding_api_key=embedding_api_key)
    # vectorstore: VectorStore = milvus_initialization(URI_link='http://localhost:19530', collection_name=collection_name, embedding_model=embedding)
    vectorstore: VectorStore = mongodb_initialization(URI_link='', collection_name=collection_name, embedding_model=embedding)
    retriever: EnsembleRetriever|BM25Retriever = get_retriever(vectorstore, collection_name)
    agent_executor = get_llm_and_agent(retriever, llm, msgs)
    logger.info(msg="Done backend initialization")

    # Setup source data chosen
    setup_data_source(llm=llm, vectorstore=vectorstore)

    # Handle user input
    handle_user_input(msgs, agent_executor)


if __name__ == "__main__":
    # main()

    from backend.vectorstore.mongodb_vectorstore import mongodb_initialization, add_files_to_mongodb
    from backend.get_model import get_embeddings_model
    from langchain_core.embeddings.embeddings import Embeddings
    from dotenv import load_dotenv
    import os
    from uuid import uuid4

    from langchain_core.documents import Document
    # Load OPENAPI KEY
    load_dotenv()
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    document_1 = Document(
    page_content="I had chocalate chip pancakes and scrambled eggs for breakfast this morning.",
    metadata={"source": "tweet"},
        )
    
    document_2 = Document(
    page_content="Building an exciting new project with LangChain - come check it out!",
    metadata={"source": "tweet"},
    )
    document_3 = Document(
    page_content="LangGraph is the best framework for building stateful, agentic applications!",
    metadata={"source": "tweet"},
    )
    
    documents = [document_1, document_2, document_3]
    uuids = [str(uuid4()) for _ in range(len(documents))]

    embedding: Embeddings | None = get_embeddings_model(embedding_choice='OpenAI', embedding_api_key=OPENAI_API_KEY)
    vector_store: VectorStore = mongodb_initialization(URI_link='', collection_name='test', embedding_model=embedding)
    # vector_store.add_documents(documents=documents, ids=uuids)
    results = vector_store.similarity_search(
    "LangChain provides abstractions to make working with LLMs easy", k=2
    )
    for res in results:
        print(f"* {res.page_content} [{res.metadata}]")
    
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 1, "score_threshold": 0.2},
    )
    response =retriever.invoke("LangChain provides abstractions to make working with LLMs easy")
    print(response)

    