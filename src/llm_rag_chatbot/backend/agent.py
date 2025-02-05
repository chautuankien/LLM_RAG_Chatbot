# Import important libraries
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_milvus import Milvus
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import create_retriever_tool
from langchain_core.vectorstores import VectorStore
from langchain.retrievers import EnsembleRetriever
from langchain.agents import create_openai_functions_agent, AgentExecutor


from src.llm_rag_chatbot.backend.vectorstore import milvus_initialization

# Load OPENAPI KEY
# load_dotenv()
# DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
# DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
 

def get_retriever(vectorstore: VectorStore, collection_name: str) -> EnsembleRetriever:
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

def get_llm_and_agent(retriever: EnsembleRetriever, model_choice: str, llm_api_key:str):
    """
    Create LLMs and Agent
    Args:
        retriever: ensemble retriever
        model_choise: LLMs model
    """
    # Initialize LLM model
    if model_choice == 'DeepSeek':
        llm = ChatOpenAI(
            temperature=0.2,
            streaming=True,
            model='deepseek-chat',
            api_key=llm_api_key,
            base_url="https://api.deepseek.com",
        )
    elif model_choice == 'OpenAI':
        llm = ChatOpenAI(
            temperature=0.2,
            streaming=True,
            model='gpt-4o-mini',
            api_key=llm_api_key,
        )
    
    
    # Initialize search tool for agent
    tool = create_retriever_tool(
        retriever=retriever,
        name='find',
        description='Search for information of Stack AI.'
    )
    tools = [tool]

    # Create prompt template for agent
    system_template = """You are an expert in AI. Your name is ChatchatAI"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Create agent
    agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)



if __name__ == "__main__":
    retriever = get_retriever()
    # agent_executor = getllm_and_agent(retriever)
    # response = agent_executor.invoke(
    #     {
    #         "input": "how can langsmith help with testing?",
    #         "chat_history": []
    #     }
    # )
    # print(response)