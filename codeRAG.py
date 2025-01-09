import getpass
import os
import bs4
os.environ['USER_AGENT'] = 'myagent' # langchain_community need website / brower variable

from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

OPENAI_API_KEY = ""
WEBSITE = "https://bwfbadminton.com/"
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
    # web_path=(WEBSITE,),
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)

docs = loader.load()
assert len(docs) == 1
print(f"Total characters: {len(docs[0].page_content)}")

# splitting documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)
print("Split the document into sub-documents", len(all_splits))

_ = vector_store.add_documents(documents=all_splits)

prompt = hub.pull("rlm/rag-prompt")

# the data to put inside the application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# the control flow
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

response = graph.invoke({"question": "What is Task Decomposition?"})
print("Answer: ", response["answer"])

# # alternative
# question = "..."

# retrieved_docs = vector_store.similarity_search(question)
# docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
# prompt = prompt.invoke({"question": question, "context": docs_content})
# answer = llm.invoke(prompt)