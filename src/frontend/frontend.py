import io
from collections.abc import Sequence
import streamlit as st

from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from unstructured.documents.elements import Element

from backend.vectorstore import add_files_to_milvus, add_data_to_milvus_url
from backend.utils.data_parser import pdf_parser, extract_and_convert_all_data

def setup_page() -> None:
    st.set_page_config(
        page_title="LLM RAG Chatbot",
        page_icon="🤖",
        layout="wide",
    )

# ==== APP INITIALIZATION ====
def initialize_app() -> None:
    setup_page()

# ==== SIDEBAR ====
def setup_api_keys() -> Sequence[str]:
    with st.sidebar:
        with st.popover(label="Setup"):
            # Choose AI Model0
            model_choice: str = st.selectbox(
                label="Choose AI Model",
                options=["DeepSeek", "OpenAI"]
            )
        
            # Choose Embedding model
            embedding_choice: str = st.selectbox(
                label="Choose Embedding model",
                options=["OpenAI"]
            )

            # Enter collection name in Milvus database
            collection_name: str = st.text_input(
                label="Type collection name need to query",
                value="data_test",
                help="Type collection name you want to use to query infomation"
            )

            if model_choice == embedding_choice == "OpenAI":
                llm_api_key: str = st.text_input(label="Enter API Key", type="password")
                embedding_api_key: str = llm_api_key
            else:
                llm_api_key: str = st.text_input(label="Enter LLM API Key", type="password")
                embedding_api_key: str = st.text_input(label="Enter Embedding API Key", type="password")
        
        if not llm_api_key or not embedding_api_key:
            st.warning(body="Please enter API Key.")
            st.stop()

    return model_choice, embedding_choice, llm_api_key, embedding_api_key, collection_name

def setup_data_source(llm: BaseLanguageModel | None, vectorstore: VectorStore) -> None:
    """
    Custom sidebar
    """
    with st.sidebar:
        # Choose Source Data
        data_source: str = st.selectbox(
            label="Choose Source Data",
            options=["Upload File", "URL"]
        )
        
        # Process Source Data
        if data_source == "Upload File":
            handle_upload_files(llm=llm, vectorstore=vectorstore)
        # else:
        #     handle_url_input(embedding=llm)
        
                              
def handle_upload_files(llm: BaseLanguageModel | None, vectorstore: VectorStore) -> None:
    """
    Handle when user choose to upload local file
    """

    uploaded_file: io.BytesIO | None = st.file_uploader(label="Upload Files", accept_multiple_files=False)
    if not uploaded_file:
        with st.spinner(text="Process data..."):
            raw_data: list[Element] | None = pdf_parser(input_data=uploaded_file)
            text_docs, image_docs, table_docs = extract_and_convert_all_data(elements=raw_data, decription_model=llm)

            try:
                add_files_to_milvus(
                    documents=[text_docs, image_docs, table_docs],
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








