import sys
import io
import os
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
MISTAKES_BOOK = "mistakes.json"

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
        Below is the content from the course. Based on this content, please create exactly 3 questions. 
        Each question can be either a multiple-choice and you must provide the correct answer.

        Important: You must ONLY output valid JSON, with the following structure (no extra text or explanation):
        {{ "subject": "Subject name", "questions": [ {{ "question": "Question text", "options": ["Option A", "Option B", "Option C", "Option D"], "answer": "Correct answer" }}, ... ] }}
        
        - Do not include any additional keys or fields.
        - Do not include any extra text outside the JSON.

        Course Content:
        {knowledge}
    """

    question = llm.invoke(prompt_Temp)
    return question.content

def loadMistakebook(subject):
    # if not os.path.exists(MISTAKES_BOOK):
    #     return []
    # else:
        with open(MISTAKES_BOOK, "r") as f:
            mistakes = f.read()
            if mistakes == "":
                return []
            else: 
                data = json.loads(mistakes)
                return data[subject]
            # random.sample(sample_questions, 2)

def saveMistakesbook(mistake, subject):
    if not os.path.exists(MISTAKES_BOOK):
         with open(MISTAKES_BOOK, "w", encoding="utf-8") as f:
            data = {}
            data[subject] = mistake
            json.dump(data, f, ensure_ascii=False, indent=2)

    else:
        with open(MISTAKES_BOOK, "r") as f:
            oldMistakes = f.read()
            if oldMistakes == "":
                return []
            else:
                data = json.loads(oldMistakes)
                data[subject] = mistake
                with open(MISTAKES_BOOK, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

def quiz(questions, subject, state):
    history_mistake = loadMistakebook(subject)
    allQuestion = history_mistake + questions['questions']
    # print(allQuestion, questions['questions'])

    state["subject"] = subject
    state["questions"] = allQuestion
    state["current_index"] = 0
    state["incorrect"] = []

    question = questions['questions'][0]
    message = "Question 1: \n" + question['question']
    if question["options"]:
        for i, opt in enumerate(question["options"]):
            message += f"\n{chr(i+65)}. {opt}"
    return message, state

def submitAns(userAns, state):
    questions = state.get("questions", [])
    idx = state.get("current_index", 0)
    incorrect_list = state.get("incorrect", [])
    # print("submit", ord(userAns.upper())-65, questions, idx, incorrect_list)

    if idx < len(questions):
        q = questions[idx]
        correct_answer = q["answer"]

        if q['options'][ord(userAns.upper())-65] == correct_answer:
            feedback = f"Correct! The answer is: {correct_answer}"
        else:
            feedback = f"Wrong! The correct answer is: {correct_answer}"
            incorrect_list.append(q)

        idx += 1
        state["current_index"] = idx
        state["incorrect"] = incorrect_list

        if idx < len(questions):
            next_q = questions[idx]
            q_msg = f"Question {idx+1}: {next_q['question']}"
            if next_q.get("options"):
                for i, opt in enumerate(next_q["options"]):
                    q_msg += f"\n{chr(i+65)}. {opt}"
            return f"{feedback}\n\n{q_msg}", state
        else:
            if len(incorrect_list) == 0:
                return f"{feedback}\n\nAll questions answered correctly! Mistakes cleared.", state
            else:
                saveMistakesbook(incorrect_list, state["subject"])
                return (
                    f"{feedback}\n\nQuiz ended. {len(incorrect_list)} mistakes saved."
                ), state
    else:
        return "No more questions. Please restart the quiz.", state



def process(loader):
    docs = loader.load()

    # print(f"Loaded {len(docs)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)

    # print(f"Split the documents into {len(all_splits)} sub-documents")

    vector_store.add_documents(documents=all_splits)

def handleInput(file, subjectName, state):
    if file and subjectName:
        process(PyPDFLoader(file))
        text = generateQuestion("Generate 3 question", subjectName)
        jsonData = json.loads(text)
        quizQuestion, state = quiz(jsonData, subjectName, state)

        # response =  "Upload Complete (PDF)"
    else:
        return "Please upload file or type the subject name", state
        
    return quizQuestion, state


# initiate the Gradio app
with gr.Blocks() as chatbot:
    state = gr.State({
            "subject": str,
            "questions": [],
            "current_index": 0,
            "incorrect": []
        })
    
    with gr.Row():
        upload_file = gr.File(label="Upload PDF")
        subject_name  = gr.Textbox(label="Subject Name", placeholder="Type the Subject", lines=2)

    upload_button = gr.Button("Upload")
    output_area = gr.Textbox(label="Question", lines=1, interactive=False)

    ans_area = gr.Textbox(label="Your Answer")
    submit_button = gr.Button("Submit Answer")

    # quiz_button = gr.Button("Start Quiz")

    upload_button.click(handleInput, inputs=[upload_file, subject_name, state], outputs=[output_area, state])
    submit_button.click(submitAns, inputs=[ans_area, state], outputs=[output_area, state])

# launch the Gradio app
chatbot.launch()