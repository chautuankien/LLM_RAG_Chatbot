from src.llm_rag_chatbot.frontend.frontend import initialize_app, setup_side_bar, setup_chat_interface, handle_user_input
from src.llm_rag_chatbot.backend.agent import get_retriever, get_llm_and_agent
from src.llm_rag_chatbot.backend.vectorstore import milvus_initialization

def main():
    # Initialize frontend
    initialize_app()
    model_choice, embedding_choice, llm_api_key, embedding_api_key, collection_name = setup_side_bar()
    msgs = setup_chat_interface()

    # Initialize backend
    vectorstore = milvus_initialization('http://localhost:19530', collection_name, embedding_choice, embedding_api_key)
    retriever = get_retriever(vectorstore, collection_name)
    agent_executor = get_llm_and_agent(retriever, model_choice, llm_api_key)
    
    # Handle user input
    handle_user_input(msgs, agent_executor)


if __name__ == "__main__":
    main()
    # msgs = setup_chat_interface("DeepSeek")
    # agent_executor = get_llm_and_agent(get_retriever(), "DeepSeek")
    # handle_user_input(msgs, agent_executor)