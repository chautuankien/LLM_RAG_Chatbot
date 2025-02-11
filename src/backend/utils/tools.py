from langchain_community.tools import DuckDuckGoSearchResults
from pydantic import BaseModel, Field

class DDGInput(BaseModel):
    query: str = Field(description="search query to look up")

def get_DuckDuckGoSearchResults_tool():
    # General internet search using DuckDuckGo
    return DuckDuckGoSearchResults(name="duck_duck_go", args_schema=DDGInput)