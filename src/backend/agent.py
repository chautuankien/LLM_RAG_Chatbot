# Import important libraries
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_milvus import Milvus
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import create_retriever_tool
from langchain_core.vectorstores import VectorStore
from langchain_core.language_models.base import BaseLanguageModel
from langchain.retrievers import EnsembleRetriever
from langchain.agents import AgentExecutor, create_react_agent, create_tool_calling_agent
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory
from langchain.memory import ConversationBufferMemory

from backend.utils.tools import get_DuckDuckGoSearchResults_tool
 

def get_retriever(vectorstore: VectorStore, collection_name: str) -> EnsembleRetriever|BM25Retriever:
    """
    Create an ensemble retriver combine vector search (dense retriver) and BM25 (sparse retriver)
    Args:
        collection_name (str): collection name for Milvus vector search
    """
    try:
        # Create  vector retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

        # Create BM25 retriver from all documents
        documents = [
            Document(page_content=doc.page_content, metadata=doc.metadata)
            for doc in vectorstore.similarity_search("", k=100)
        ]

        if not documents:
            raise ValueError(f"Cannot find documents in collection: {collection_name}")
        
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 4

        # Initialize ensemble retriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, retriever],
            weights=[0.3, 0.7]
        )

        return ensemble_retriever

    except Exception as e:
        print(f"Error when creating emsemble: {str(e)}")
        # Return default retriever
        default_doc = [
            Document(
                page_content="There is an error occur when connecting to the database. Please try again",
                meta_data={"source": "error"}
            )
        ]
        return BM25Retriever.from_documents(default_doc)

def get_llm_and_agent(retriever: EnsembleRetriever|BM25Retriever, llm_model: BaseLanguageModel, msgs: StreamlitChatMessageHistory):
    """
    Create LLMs and Agent
    Args:
        retriever: ensemble retriever
        model_choise: LLMs model
    """  
    memory = ConversationBufferMemory(
        chat_memory=msgs,
        return_messages=True,
        memory_key="chat_history",
        output_key="output",
    )

    # Initialize search tool for agent
    retriever_tool = create_retriever_tool(
        retriever=retriever,
        name='find',
        description='Search tool'
    )
    search_tool = get_DuckDuckGoSearchResults_tool()
    tools = [retriever_tool, search_tool]

    # Create prompt template for agent
    system_template = """You are an expert in AI. Your name is ChatchatAI"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Create agent
    agent = create_tool_calling_agent(llm=llm_model, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, memory=memory, return_intermediate_steps=True, verbose=True)
