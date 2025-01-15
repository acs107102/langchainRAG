
# os.environ['USER_AGENT'] = 'myagent' # langchain_community need website / brower variable
# os.environ['USER_AGENT'] = '"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"'
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from dotenv import load_dotenv
load_dotenv()

DATA_PATH = "Data"
CHROMA_PATH = "Chroma_DB"

# initiate the embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# initiate the vector store
vector_store = Chroma(
    collection_name="vector_collection",
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH,
)

loader = PyPDFDirectoryLoader(DATA_PATH)

docs = loader.load()

# assert len(docs) == 1
print(len(docs))
print(docs)
# print(f"Total characters: {len(docs[0].page_content)}")
# print(docs[0].page_content)

# splitting documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)
print("Split the document into sub-documents", len(all_splits))

_ = vector_store.add_documents(documents=all_splits)

# # query analysis
# total_documents = len(all_splits)
# third = total_documents // 3

# for i, document in enumerate(all_splits):
#     if i < third:
#         document.metadata["section"] = "beginning"
#     elif i < 2 * third:
#         document.metadata["section"] = "middle"
#     else:
#         document.metadata["section"] = "end"

# # storing documents - index chunks
# _ = vector_store.add_documents(documents=all_splits)

# # check langchain prompt hub
# prompt = hub.pull("rlm/rag-prompt")

# # schema for search query
# class Search(TypedDict):
#     """Search query."""

#     query: Annotated[str, ..., "Search query to run."]
#     section: Annotated[
#         Literal["beginning", "middle", "end"],
#         ...,
#         "Section to query.",
#     ]

# # the data to put inside the application
# class State(TypedDict):
#     question: str
#     query: Search
#     context: List[Document]
#     answer: str

# def analyze_query(state: State):
#     structured_llm = llm.with_structured_output(Search)
#     query = structured_llm.invoke(state["question"])
#     return {"query": query}

# def retrieve(state: State):
#     query = state["query"]
#     retrieved_docs = vector_store.similarity_search(
#         query["query"],
#         filter=lambda doc: doc.metadata.get("section") == query["section"],
#     )
#     return {"context": retrieved_docs}


# def generate(state: State):
#     docs_content = "\n\n".join(doc.page_content for doc in state["context"])
#     messages = prompt.invoke({"question": state["question"], "context": docs_content})
#     response = llm.invoke(messages)
#     return {"answer": response.content}

# # the control flow
# graph_builder = StateGraph(State).add_sequence([analyze_query, retrieve, generate])
# graph_builder.add_edge(START, "analyze_query")
# graph = graph_builder.compile()

# # question
# QUESTION = "Who is number one in woman single badminton in 2020?"
# response = graph.invoke({"question": QUESTION})
# print("Answer: ", response["answer"])