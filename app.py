from src.llm_rag_chatbot.backend.agent import get_retriever, get_llm_and_agent
from src.llm_rag_chatbot.frontend.frontend import initialize_app, setup_side_bar, setup_chat_interface, handle_user_input

def main():
    initialize_app()
    model_choice, embedding_choice, api_key, collection_to_query = setup_side_bar()
    msgs = setup_chat_interface()

    retriever = get_retriever(collection_to_query, embedding_choice)
    agent_executor = get_llm_and_agent(retriever, model_choice, api_key)
    
    handle_user_input(msgs, agent_executor)


if __name__ == "__main__":
    main()
    # msgs = setup_chat_interface("DeepSeek")
    # agent_executor = get_llm_and_agent(get_retriever(), "DeepSeek")
    # handle_user_input(msgs, agent_executor)