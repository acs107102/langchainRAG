
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