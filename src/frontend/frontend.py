from collections.abc import Sequence
from dotenv import load_dotenv
import logging.config
from functools import partial

import streamlit as st
from streamlit_feedback import streamlit_feedback
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.vectorstores import VectorStore
from langchain_core.runnables import RunnableConfig
from langchain_core.tracers.context import tracing_v2_enabled
from langsmith import Client, traceable

from frontend.frontend_utils import handle_upload_files

logger = logging.getLogger(name="frontend")

client = Client()

def setup_page() -> None:
    st.set_page_config(
        page_title="LLM RAG Chatbot",
        page_icon="🤖",
        layout="wide",
    )

# ==== APP INITIALIZATION ====
def initialize_app() -> None:
    setup_page()
    load_dotenv()

    if "feedback_key" not in st.session_state:
        st.session_state.feedback_key = 0

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
        
                            
# ==== MAIN INTERFACE ====
def _submit_feedback(user_response: dict, emoji=None, run_id=None):
    score = {"👍": 1, "👎": 0}.get(user_response.get("score"))
    client.create_feedback(
        run_id=run_id,
        key=user_response["type"],
        score=score,
        comment=user_response.get("text"),
        value=user_response.get("score"),
    )
    return user_response

def setup_chat_interface():
    st.title("🤖 LLM RAG Chatbot")
    
    msgs = StreamlitChatMessageHistory(key="langchain_messages")

    if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
        msgs.clear()
        st.session_state.immediate_steps = {}
        # msgs.add_ai_message("Hello! How can I help you today?")
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello! How can I help you today?"}
            ]
            msgs.add_ai_message("Hello! How can I help you today?")
    
    # for msg in st.session_state.messages:
    #     role = "assistant" if msg["role"] == "assistant" else "human"
    #     st.chat_message(role).write(msg["content"])

    feedback_kwargs = {
        "feedback_type": "thumbs",
        "optional_text_label": "Rate this response in LangSmith",
    }
    
    avatars = {"human": "user", "ai": "assistant"}
    for idx, msg in enumerate(msgs.messages):
        with st.chat_message(avatars[msg.type]):
            # Render intermediate steps if any were saved
            for step in st.session_state.immediate_steps.get(str(idx), []):
                if step[0].tool == "_Exception":
                    continue
                with st.status(f"**{step[0].tool}**: {step[0].tool_input}", state="complete"):
                    st.write(step[0].log)
                    st.write(step[1])
            st.write(msg.content)
        
        if msg.type == "ai":
            feedback_key = f"feedback_{int(idx/2)}"

            if feedback_key not in st.session_state:
                st.session_state[feedback_key] = None
            if f"run_{int(idx/2)}" not in st.session_state:
                st.session_state[f"run_{int(idx/2)}"] = 0

            disable_with_score = (
                st.session_state[feedback_key].get("score")
                if st.session_state[feedback_key]
                else None
            )
            # This actually commits the feedback
            streamlit_feedback(
                **feedback_kwargs,
                key=feedback_key,
                disable_with_score=disable_with_score,
                on_submit=partial(
                    _submit_feedback, run_id=st.session_state[f"run_{int(idx/2)}"]
                ),
            )
    
    return msgs

def handle_user_input(msgs, agent_executor):
    """
    Handle user input
    1. Show user message
    2. Get response from AI model, and show AI message
    3. Add response to chat history
    """

    if st.session_state.get("run_url"):
        st.markdown(
            f"View trace in [🦜🛠️ LangSmith]({st.session_state.run_url})",
            unsafe_allow_html=True,
        )

    if prompt := st.chat_input("Ask me anything..."):
        # Save and show user message
        st.session_state.messages.append({"role":"human", "content": prompt})
        st.chat_message("human").write(prompt)
        msgs.add_user_message(prompt)

        # Process user input and show AI response
        
        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container())
            cfg = RunnableConfig()
            cfg["callbacks"] = [st_callback]
            
            # Get chat history
            chat_history = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in st.session_state.messages[:-1]
            ]
            with tracing_v2_enabled("langsmith-streamlit-agent") as cb:
                # Get AI response
                response = agent_executor.invoke(
                    {
                        "input": prompt,
                        "chat_history": chat_history
                    },
                    cfg
                )

                feedback_kwargs = {
                    "feedback_type": "thumbs",
                    "optional_text_label": "Please provide extra information",
                    "on_submit": _submit_feedback,
                    }
                run = cb.latest_run
                feedback_index = int(
                (len(st.session_state.get("langchain_messages", [])) - 1) / 2
                )
                st.session_state[f"run_{feedback_index}"] = run.id

                # Save and show AI response
                # st.session_state.messages.append({"role":"assistant", "content": output})
                st.write(response["output"])
                st.session_state.immediate_steps[str(len(msgs.messages) - 1)] = response["intermediate_steps"]
                # logger.debug(msg=f"LLM response: {response}")
                # logger.debug(msg=f"Chat history: {msgs.messages}")
                # msgs.add_ai_message(output)

                # This displays the feedback widget and saves to session state
                # It will be logged on next render
                streamlit_feedback(**feedback_kwargs, key=f"feedback_{feedback_index}")
                try:
                    url = cb.get_run_url()
                    st.session_state.run_url = url
                    st.markdown(
                        f"View trace in [🦜🛠️ LangSmith]({url})",
                        unsafe_allow_html=True,
                    )
                except Exception:
                    logger.exception("Failed to get run URL.")

# if __name__ == "__main__":
#     msgs = setup_chat_interface("DeepSeek")
#     agent_executor = get_llm_and_agent(get_retriever(), "DeepSeek")
#     handle_user_input(msgs, agent_executor)








