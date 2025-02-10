from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from langchain_core.language_models.base import BaseLanguageModel

import streamlit as st
from pydantic import SecretStr

@st.cache_resource
def get_embeddings_model(embedding_choice: str, embedding_api_key: str) -> Embeddings | None:
    """
    Retrieves the embeddings model based on the specified choice and API key.
    Args:
        embedding_choice (str): The choice of embeddings model to use. Currently supports "OpenAI".
        embedding_api_key (str): The API key required to access the embeddings model.
    Returns:
        Embeddings: The embeddings model instance if successfully loaded, otherwise None.
    Raises:
        Exception: If there is an error loading the embeddings model, an error message is displayed and None is returned.
    """
    try:
        if embedding_choice == "OpenAI":
            embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=SecretStr(secret_value=embedding_api_key))
        
            return embeddings
    except Exception as e:
        st.error(f"Error loading embeddings model: {e}")
        return None

@st.cache_resource
def get_llm_model(model_choice: str, llm_api_key: str) -> BaseLanguageModel | None:
    """
    Retrieve a language model instance based on the specified model choice.
    Args:
        model_choice (str): The choice of language model to use. Options are 'DeepSeek' or 'OpenAI'.
        llm_api_key (str): The API key required to access the language model service.
    Returns:
        BaseLanguageModel: An instance of the selected language model, otherwise None.
    Raises:
        ValueError: If an unsupported model_choice is provided.
    """
    try:
        if model_choice == 'DeepSeek':
            
            llm = ChatOpenAI(
                temperature=0.2,
                streaming=True,
                model='deepseek-chat',
                api_key=SecretStr(secret_value=llm_api_key),
                base_url="https://api.deepseek.com",
            )
            return llm
        
        elif model_choice == 'OpenAI':
            llm = ChatOpenAI(
                temperature=0.2,
                streaming=True,
                model='gpt-4o-mini',
                api_key=SecretStr(secret_value=llm_api_key),
            )
            return llm
    
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
    
    