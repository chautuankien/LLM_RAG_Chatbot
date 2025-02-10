import json
import logging.config
import logging.handlers
import pathlib

from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory
from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.vectorstores.base import VectorStore

from frontend.frontend import initialize_app, setup_api_keys, setup_chat_interface, setup_data_source, handle_user_input

from backend.get_model import get_embeddings_model, get_llm_model
from backend.vectorstore import milvus_initialization
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
    vectorstore: VectorStore = milvus_initialization(URI_link='http://localhost:19530', collection_name=collection_name, embedding_model=embedding)
    retriever = get_retriever(vectorstore, collection_name)
    agent_executor = get_llm_and_agent(retriever, model_choice, llm_api_key)
    logger.info(msg="Done backend initialization")

    # Setup source data chosen
    setup_data_source(llm=llm, vectorstore=vectorstore)

    # Handle user input
    handle_user_input(msgs, agent_executor)


if __name__ == "__main__":
    main()
    # msgs = setup_chat_interface("DeepSeek")
    # agent_executor = get_llm_and_agent(get_retriever(), "DeepSeek")
    # handle_user_input(msgs, agent_executor)