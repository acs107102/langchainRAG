import sys
import io
import os
import json
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from dotenv import load_dotenv
load_dotenv()
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

DATA_PATH = "School"
CHROMA_PATH = "Chroma_DB_School"
SUBJECT_LIST_FILE = "subject.json"

# initiate the embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# initiate the vector store
vector_store = Chroma(
    collection_name="vector_collection",
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH,
)

folder = os.listdir(DATA_PATH)
print(folder)

with open(SUBJECT_LIST_FILE, "w", encoding="utf-8") as f:
    json.dump(folder, f)

for name in folder:
    print(name)
    loader = PyPDFDirectoryLoader(DATA_PATH + "/" + name)
    docs = loader.load()

    for d in docs:
        d.metadata["subject"] = name

    # splitting documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)
    print("Split the document into sub-documents", len(all_splits))

    _ = vector_store.add_documents(documents=all_splits)