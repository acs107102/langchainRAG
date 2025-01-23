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
WEBSITE = "https://en.wikipedia.org/wiki/"

llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# vector database : embedding
vector_store = Chroma(
    collection_name="vector_collection",
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH,
)

# retrieve
num = 5
retriever = vector_store.as_retriever(search_kwargs={'k': num})

def streamResponse(message, history):
    print(f"Input: {message}. History: {history}\n")
    '''
    {'role': 'user', 'metadata': {'title': None, 'id': None, 'parent_id': None, 'duration': None, 'status': None}, 'content': 'what is banana', 'options': None}, {'role': 'assistant', 'metadata': {'title': None, 'id': None, 'parent_id': None, 'duration': None, 'status': None}, 'content': 'A banana is an elongated, edible fruit that is botanically classified as a berry. It is produced by several kinds of large treelike herbaceous flowering plants in the genus Musa. In various regions, cooking bananas are referred to as plantains, which differentiates them from dessert bananas. The fruit can vary in size, color, and firmness but is typically elongated and curved, with soft, starchy flesh covered by a peel that can have different colors when ripe. Bananas grow in clusters near the top of the plant. Most modern edible seedless cultivated bananas are derived from two wild species: Musa acuminata and Musa balbisiana, or hybrids of these species.', 'options': None}
    '''

    # find k documents
    docs = retriever.invoke(message)
    '''Document(id='ac3d3064-0815-4fe3-ab52-2a31e4ba711c', metadata={'language': 'en', 'source': 'https://en.wikipedia.org/wiki/apple\n', 'title': 'Bad title - Wikipedia'}, page_content='In other projects\n\t\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nAppearance\nmove to sidebar\nhide\n\n\n\n\n\n\n\n\n\n\nThe requested page title contains unsupported characters: "\n".\nReturn to Main Page.\n\nRetrieved from "https://en.wikipedia.org/wiki/Special:Badtitle"\n\n\n\n\n\n\n\n\n\nPrivacy policy\nAbout Wikipedia\nDisclaimers\nContact Wikipedia\nCode of Conduct\nDevelopers\nStatistics\nCookie statement\nMobile view\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nSearch\n\n\n\n\n\n\n\n\n\n\n\n\n\nSearch\n\n\n\n\n\nBad title\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nAdd topic')'''

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

        # print("rag_prompt", rag_prompt)

        # stream the response to the Gradio App
        for response in llm.stream(rag_prompt):
            partial_message += response.content
            yield partial_message
    
def process(loader):
    docs = loader.load()

    # print(f"Loaded {len(docs)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)

    # print(f"Split the documents into {len(all_splits)} sub-documents")

    vector_store.add_documents(documents=all_splits)

def handle_input(file, fileName, link):
        if file:
            process(PyPDFLoader(file))
            response =  "Upload Complete (PDF)"
        elif fileName:
            loader = WebBaseLoader(
                web_paths=(WEBSITE+fileName,),
            )
            process(loader)
            response = f"Upload Complete (Wiki: {fileName})"
        elif link:
            loader = WebBaseLoader(
                web_paths=(link,),
            )
            process(loader)
            response = f"Upload Complete (Link: {link})"
        else:
            return "Please upload file", gr.update(visible=False)
        
        return response, gr.update(visible=True)


# initiate the Gradio app
with gr.Blocks() as chatbot:
    state = gr.State()
    with gr.Row():
        upload_button = gr.File(label="Upload PDF")
        with gr.Column():
            wiki_search  = gr.Textbox(label="Wiki Search", placeholder="Type anything you want to know from wiki", lines=2)
            website_search = gr.Textbox(label="Website Search", placeholder="Type any links", lines=2)

    submit_button = gr.Button("Submit")
    output_area = gr.Textbox(label="System Message", lines=1, interactive=False)

    with gr.Column(visible=False) as chat_row:

        gr.Markdown("### Chat with the LLM")
        gr.ChatInterface(
            streamResponse,
            type="messages")

    submit_button.click(handle_input, inputs=[upload_button, wiki_search, website_search], outputs=[output_area, chat_row])

# launch the Gradio app
chatbot.launch()