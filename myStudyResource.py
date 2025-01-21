import sys
import io
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import gradio as gr
import json

from dotenv import load_dotenv
load_dotenv()
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

CHROMA_PATH = "Chroma_DB_School"
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
    
def generateQuestion(message, subject):
    # PromptTemplate.from_template(
        
    # )
    docs = retriever.invoke(message)

    knowledge = ""
    for doc in docs:
        knowledge += doc.page_content+"\n\n"

    prompt_Temp = f"""
        You are a teacher. The subject is: {subject}
        Below is the content from the course. Based on this content, please create exactly 10 questions. 
        Each question can be either a multiple-choice or a short-answer question, and you must provide the correct answer.

        Important: You must ONLY output valid JSON, with the following structure (no extra text or explanation):
        {{ "subject": "Subject name", "questions": [ {{ "question": "Question text", "options": ["Option A", "Option B", "Option C"], "answer": "Correct answer" }}, ... ] }}
        

        - If the question is short-answer, the "options" field can be an empty array (e.g., `"options": []`) or omitted.
        - Do not include any additional keys or fields.
        - Do not include any extra text outside the JSON.

        Course Content:
        {knowledge}
    """

    question = llm.invoke(prompt_Temp)
    print(question.content)
    return question.content
    

def process(loader):
    docs = loader.load()

    # print(f"Loaded {len(docs)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)

    # print(f"Split the documents into {len(all_splits)} sub-documents")

    vector_store.add_documents(documents=all_splits)

def handle_input(file, subjectName):
        if file and subjectName:
            process(PyPDFLoader(file))
            text = generateQuestion("Generate 5 question", subjectName)
            data = json.loads(text)
            # response =  "Upload Complete (PDF)"
        else:
            return "Please upload file or type the subject name"
        
        return text


# initiate the Gradio app
with gr.Blocks() as chatbot:
    state = gr.State()
    with gr.Row():
        upload_button = gr.File(label="Upload PDF")
        subject_name  = gr.Textbox(label="Subject Name", placeholder="Type the Subject", lines=2)

    submit_button = gr.Button("Submit")
    output_area = gr.Textbox(label="System Message", lines=1, interactive=False)



    submit_button.click(handle_input, inputs=[upload_button, subject_name], outputs=[output_area])

# launch the Gradio app
chatbot.launch()