import getpass
import os
import bs4
import sys
import io
os.environ['USER_AGENT'] = 'myagent' # langchain_community need website / brower variable
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from langchain import hub
from langchain_community.document_loaders import WebBaseLoader, WikipediaLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph, MessagesState
from typing_extensions import List, TypedDict
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from typing import Literal
from typing_extensions import Annotated

OPENAI_API_KEY = ""
WEBSITE = "https://en.wikipedia.org/wiki/BWF_World_Ranking"
LANGCHAIN_API_KEY = ""

# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = InMemoryVectorStore(embeddings)

# loading documents
loader = WebBaseLoader(
    web_paths=(WEBSITE,),
)

docs = loader.load()
assert len(docs) == 1
print(len(docs))
print(f"Total characters: {len(docs[0].page_content)}")

# splitting documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)
print("Split the document into sub-documents", len(all_splits))

# query analysis
total_documents = len(all_splits)
third = total_documents // 3

for i, document in enumerate(all_splits):
    if i < third:
        document.metadata["section"] = "beginning"
    elif i < 2 * third:
        document.metadata["section"] = "middle"
    else:
        document.metadata["section"] = "end"

# storing documents - index chunks
_ = vector_store.add_documents(documents=all_splits)

# check langchain prompt hub
prompt = hub.pull("rlm/rag-prompt")

# schema for search query
class Search(TypedDict):
    """Search query."""

    query: Annotated[str, ..., "Search query to run."]
    section: Annotated[
        Literal["beginning", "middle", "end"],
        ...,
        "Section to query.",
    ]

# the data to put inside the application
class State(TypedDict):
    question: str
    query: Search
    context: List[Document]
    answer: str

def analyze_query(state: State):
    structured_llm = llm.with_structured_output(Search)
    query = structured_llm.invoke(state["question"])
    return {"query": query}

def retrieve(state: State):
    query = state["query"]
    retrieved_docs = vector_store.similarity_search(
        query["query"],
        filter=lambda doc: doc.metadata.get("section") == query["section"],
    )
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# the control flow
graph_builder = StateGraph(State).add_sequence([analyze_query, retrieve, generate])
graph_builder.add_edge(START, "analyze_query")
graph = graph_builder.compile()

# question
QUESTION = "Who is rank one in woman single?"
response = graph.invoke({"question": QUESTION})
print("Answer: ", response["answer"])