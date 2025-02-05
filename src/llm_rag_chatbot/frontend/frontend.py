import streamlit as st
from dotenv import load_dotenv, find_dotenv

from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

from src.llm_rag_chatbot.backend.vectorstore import add_data_to_milvus_local, add_data_to_milvus_url
from src.llm_rag_chatbot.backend.data_parser import pdf_parser

def setup_page():
    st.set_page_config(
        page_title="LLM RAG Chatbot",
        page_icon="🤖",
        layout="wide",
    )

# ==== APP INITIALIZATION ====
def initialize_app():
    load_dotenv(find_dotenv())
    setup_page()

# ==== SIDEBAR ====
def setup_side_bar():
    """
    Custom sidebar
    """
    with st.sidebar:
        with st.popover("Setup"):
            # Choose AI Model0
            model_choice = st.selectbox(
                "Choose AI Model",
                ["DeepSeek", "OpenAI"]
            )
        
            # Choose Embedding model
            embedding_choice = st.selectbox(
                "Choose Embedding model",
                ["OpenAI"]
            )

            if model_choice == embedding_choice == "OpenAI":
                llm_api_key = embedding_api_key = st.text_input("Enter API Key", type="password")
            else:
                llm_api_key = st.text_input("Enter LLM API Key", type="password")
                embedding_api_key = st.text_input("Enter Embedding API Key", type="password")
        
        if not llm_api_key or not embedding_api_key:
            st.warning("Please enter API Key.")
            st.stop()

        # Choose Source Data
        data_source = st.selectbox(
            "Choose Source Data",
            ["Upload File", "URL"]
        )
        
        # Process Source Data based on Embedding choice
        if data_source == "Upload File":
            handle_upload_file(embedding_choice=embedding_choice)
        else:
            handle_url_input(embedding_choice=embedding_choice)
        
        # Add collection to query
        st.header("Collection to query")
        collection_name = st.text_input(
            "Type collection name need to query",
            "data_test",
            help="Type collection name you want to use to query infomation"
        )

        return model_choice, embedding_choice, llm_api_key, embedding_api_key, collection_name
                              
def handle_upload_file(embedding_choice: str):
    """
    Handle when user choose to upload local file
    """
    collection_name = st.text_input(
        "Collection name in database",
        "data_test",
        help="Type collection name you want to use to store in database",
    )

    dir_path = st.text_input(
        "Directory containing PDF files", 
        "data",
        help="Type directory path where the PDF files are stored",
    )

    uploaded_file = st.file_uploader("Upload Files", accept_multiple_files=False)
    if not uploaded_file:
        raw_data = pdf_parser(uploaded_file)

        with st.spinner("Loading data..."):
            try:
                add_data_to_milvus_local(
                    dir_path,
                    'http://localhost:19530', 
                    collection_name,
                    embedding_choice=embedding_choice
                )
                st.success("Data loaded successfully!")
            except Exception as e:
                st.error(f"Error loading data: {e}")
   

def handle_url_input(embedding_choice: str):
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


# ==== MAIN INTERFACE ====
def setup_chat_interface():
    st.title("🤖 LLM RAG Chatbot")
    
    msgs = StreamlitChatMessageHistory(key="langchain_messages")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! How can I help you today?"}
        ]
        msgs.add_ai_message("Hello! How can I help you today?")
    
    for msg in st.session_state.messages:
        role = "assistant" if msg["role"] == "assistant" else "human"
        st.chat_message(role).write(msg["content"])
    
    return msgs

def handle_user_input(msgs, agent_executor):
    """
    Handle user input
    1. Show user message
    2. Get response from AI model, and show AI message
    3. Add response to chat history
    """

    if prompt := st.chat_input("Ask me anything..."):
        # Save and show user message
        st.session_state.messages.append({"role":"human", "content": prompt})
        st.chat_message("human").write(prompt)
        msgs.add_user_message(prompt)

        # Process user input and show AI response
        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container())
            
            # Get chat history
            chat_history = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in st.session_state.messages[:-1]
            ]

            # Get AI response
            response = agent_executor.invoke(
                {
                    "input": prompt,
                    "chat_history": chat_history
                },
                {"callback": [st_callback]}
            )

            # Save and show AI response
            output = response["output"]
            st.session_state.messages.append({"role":"assistant", "content": output})
            st.write(output)
            msgs.add_ai_message(output)

# if __name__ == "__main__":
#     msgs = setup_chat_interface("DeepSeek")
#     agent_executor = get_llm_and_agent(get_retriever(), "DeepSeek")
#     handle_user_input(msgs, agent_executor)








