import sys
import io
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import gradio as gr

from dotenv import load_dotenv
load_dotenv()
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

DATA_PATH = "Data"
CHROMA_PATH = "Chroma_DB"
WEBSITE = "https://en.wikipedia.org/wiki"

llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# vector database : embedding
vector_store = Chroma(
    collection_name="vector_collection",
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH,
)

num = 5
retriever = vector_store.as_retriever(search_kwargs={'k': num})

def streamResponse(message, history):
    print(f"Input: {message}. History: {history}\n")

    docs = retriever.invoke(message)

    knowledge = ""

    for doc in docs:
        knowledge += doc.page_content+"\n\n"


    if message is not None:

        partial_message = ""

        rag_prompt = f"""
        You are an assistent which answers questions based on knowledge.
        While answering, you don't use your internal knowledge, 
        but solely the information in the "The knowledge" section.
        You don't mention anything to the user about the povided knowledge.

        The question: {message}

        Conversation history: {history}

        The knowledge: {knowledge}

        """

        #print(rag_prompt)

        # stream the response to the Gradio App
        for response in llm.stream(rag_prompt):
            partial_message += response.content
            yield partial_message
    
    #     history.append({"role": "user", "content": message})
    #     history.append({"role": "assistant", "content": partial_message})
    #     print(partial_message)
    # return "", history

def process_wiki(fileName):
    # loading documents
    loader = WebBaseLoader(
        web_paths=(WEBSITE+fileName,),
    )
    docs = loader.load()

    # splitting documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)
    print("Split the document into sub-documents", len(all_splits))

    vector_store.add_documents(documents=all_splits)

def process_pdf(file):
    loader = PyPDFLoader(file)
    docs = loader.load()

    print(f"Loaded {len(docs)} documents from {file}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)

    print(f"Split the documents into {len(all_splits)} sub-documents")

    vector_store.add_documents(documents=all_splits)

def handle_input(file, fileName):
        if file:
            process_pdf(file)
            return "Upload Complete (PDF)", gr.update(visible=True)
        elif fileName:
            process_wiki(fileName)
            return "Upload Complete (WIKI)",  gr.update(visible=True)
        else:
            return "Please upload file", gr.update(visible=False)


# initiate the Gradio app
with gr.Blocks() as chatbot:
    with gr.Row():
        upload_button = gr.File(label="Upload PDF")
        wiki_search  = gr.Textbox(label="Wiki Search", placeholder="Type the anything you want to know from wiki", lines=4)

    submit_button = gr.Button("Submit")
    output_area = gr.Textbox(label="Message", lines=1, interactive=False)

    with gr.Column(visible=False) as chat_row:
    #     cb = gr.Chatbot(type="messages",
    #             container=False,
    #             autoscroll=True,
    #             scale=7)
    #     msg = gr.Textbox(placeholder="Send to the LLM...",)
    #     clear = gr.ClearButton([msg, cb])
        
    # msg.submit(streamResponse, [msg, cb], [msg, cb])

        gr.Markdown("### Chat with the LLM")
        gr.ChatInterface(
            streamResponse,
            type="messages")

    submit_button.click(handle_input, inputs=[upload_button, wiki_search], outputs=[output_area, chat_row])

# launch the Gradio app
chatbot.launch()