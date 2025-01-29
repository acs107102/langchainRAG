import sys
import io
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
import gradio as gr
import json

from dotenv import load_dotenv
load_dotenv()
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

CHROMA_PATH = "Chroma_DB_School"
MISTAKES_BOOK = "mistakes.json"
SUBJECT_LIST_FILE = "subject.json"

with open(SUBJECT_LIST_FILE, "r", encoding="utf-8") as f:
    subject_list = json.load(f)

llm = ChatOpenAI(temperature=1.0, model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# vector database : embedding
vector_store = Chroma(
    collection_name="vector_collection",
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH,
)

def addSubject(subject, addNew):
    if subject == "Add New":
        if addNew not in subject_list:
            subject_list.append(addNew.strip())
            with open(SUBJECT_LIST_FILE, "w", encoding="utf-8") as f:
                subject_list.sort()
                json.dump(subject_list, f)
                return "Success add a new subject", gr.update(visible=True), gr.update(choices=['Add New'] + subject_list, value='Add New')
        elif addNew in subject_list:
            return "This subject is already in the system", gr.update(visible=True), gr.update(value='Add New')
        else:
            return "Please Type the subject", gr.update(visible=True), gr.update(value='Add New')
    else:
        return "Are you try to add new subject ? Please select \"add new\"", gr.update(visible=True), gr.update(value='Add New')

def process(loader, subjectName):
    docs = loader.load()

    # print(f"Loaded {len(docs)} documents")

    for d in docs:
        d.metadata["subject"] = subjectName

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)

    # print(f"Split the documents into {len(all_splits)} sub-documents")

    vector_store.add_documents(documents=all_splits)

def uploadFile(file, subjectName):
    if file and subjectName and subjectName != "Add New":
        process(PyPDFLoader(file), subjectName)
        return  "Upload Complete (PDF)", gr.update(visible=True)
    else:
        return "Please upload file or type the subject name", gr.update(visible=True)

def startChatbot():
    return gr.update(visible=True), gr.update(visible=False)

def subjectNameCheck(subject):
    results = vector_store.similarity_search(
        query="",
        k=1, 
        filter={"subject": subject}
    )
    return len(results) <= 0
    
def generateQuestion(subject):
    # retrieve
    num = 5
    name = str
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            'k': num,
            'filter': {"subject": subject}
            }
    )
    name = subject

    docs = retriever.invoke("Generate questions from " + name)

    knowledge = ""
    for doc in docs:
        knowledge += doc.page_content+"\n\n"

    prompt_Temp = """
        You are a teacher. The subject is: {subjectName}
        Below is the content from the course. Based on this content, please create exactly 3 multiple-choice questions.  
        Each question must focus on core theories, technical details, or key concepts discussed in the course. Avoid including:  
        - Information related to assignments, syllabus outlines, or non-critical content.  
        - Any references to authors, affiliations, citations, or metadata from research papers.

        Important Instructions:  
        1. For each question, provide exactly 4 options: Option A, Option B, Option C, and Option D.  
        2. The correct answer must be **identical** to one of the provided options.  
        3. Ensure the options are plausible distractors but only one is correct.  
        4. Strictly format the output as valid JSON with the following structure (no extra text or explanation):  

        {{ "subject": "Subject name", "questions": [ {{ "question": "Question text", "options": ["Option A", "Option B", "Option C", "Option D"], "answer": "Correct answer" }}, ... ] }}
        
        - Correct Answer Requirement: Always copy the correct answer directly from the list of options to avoid mismatches.
        - Do not include any additional keys or fields.
        - Do not include any extra text outside the JSON.

        Course Content:
        {content}
    """

    prompt_template = PromptTemplate.from_template(prompt_Temp)
    messages = prompt_template.invoke({"subjectName": subject, "content": knowledge})

    question = llm.invoke(messages)
    return question.content

def startQuiz(subjectName, state):
    if subjectName == "Add New":
        return "Please type different subject.", state, gr.update(visible=True), gr.update(visible=False)
    elif subjectNameCheck(subjectName):
        return "There is no data in this subject yet. Please upload the file.", state, gr.update(visible=True), gr.update(visible=False)
    text = generateQuestion(subjectName)
    jsonData = json.loads(text)
    quizQuestion, state = quiz(jsonData, subjectName, state)
    return quizQuestion, state, gr.update(visible=True), gr.update(visible=False)

def loadMistakebook(subject):
    if not os.path.exists(MISTAKES_BOOK):
        return []
    else:
        with open(MISTAKES_BOOK, "r", encoding="utf-8") as f:
            mistakes = f.read()
            if mistakes == "":
                return []
            else: 
                data = json.loads(mistakes)
                return data.get(subject, "NoDB")
            # random.sample(sample_questions, 2)

def saveMistakesbook(mistake, subject):
    data = {}
    if not os.path.exists(MISTAKES_BOOK):
         with open(MISTAKES_BOOK, "w", encoding="utf-8") as f:
            data[subject] = mistake
            json.dump(data, f, ensure_ascii=False, indent=2)

    else:
        with open(MISTAKES_BOOK, "r", encoding="utf-8") as f:
            oldMistakes = f.read()
            if oldMistakes != "":
                data = json.loads(oldMistakes)
            data[subject] = mistake
            with open(MISTAKES_BOOK, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

def quiz(questions, subject, state, type=True):
    # print(allQuestion, questions['questions'])
    if type:
        questionType = questions['questions']
    else:
        item = loadMistakebook(subject)
        if item == "NoDB":
            return "No data in this subject", state
        elif len(item) > 5:
            questionType = item[:5]
            saveMistakesbook(item[5:], subject)
        elif(len(item) == 0):
            return "No mistakes. Well Done : )", state
        else:
            questionType = item
            saveMistakesbook([], subject)

    state["subject"] = subject
    state["questions"] = questionType
    state["currentIdx"] = 0
    state["incorrect"] = []

    question = questionType[0]
    message = "Question 1: \n" + question['question']
    if question["options"]:
        for i, opt in enumerate(question["options"]):
            message += f"\n{chr(i+65)}. {opt}"
    return message, state

def reviewQuiz(subjectName, state):
    if subjectName == "Add New":
        return "Please type different subject.", state, gr.update(visible=True), gr.update(visible=False)
    quizQuestion, state = quiz([], subjectName, state, False)
    return quizQuestion, state, gr.update(visible=True), gr.update(visible=False)

def submitAns(userAns, state):
    questions = state.get("questions", [])
    idx = state.get("currentIdx", 0)
    incorrect_list = state.get("incorrect", [])
    # print("submit", ord(userAns.upper())-65, questions, idx, incorrect_list)

    if idx < len(questions):
        q = questions[idx]
        correctAnswer = q["answer"]
        ansIdx = ord(userAns[0].upper())-65
        if q['options'][ansIdx] == correctAnswer:
            feedback = f"Correct! The answer is: {correctAnswer}"
        else:
            feedback = f"Wrong! The correct answer is: {correctAnswer}"
            incorrect_list.append(q)

        idx += 1
        state["currentIdx"] = idx
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
                oldMistake = loadMistakebook(state["subject"])
                if oldMistake == "NoDB":
                    newMistake = incorrect_list
                else:
                    newMistake = oldMistake + incorrect_list
                saveMistakesbook(newMistake, state["subject"])
                return (
                    f"{feedback}\n\nQuiz ended. {len(incorrect_list)} mistakes saved."
                ), state
    else:
        return "No more questions. Please restart the quiz.", state

def streamResponse(message, history, subject):
    # retrieve
    num = 5
    if not subjectNameCheck(subject):
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                'k': num,
                'filter': {"subject": subject}
                }
        )
    else:
        message = None

    
    if message is not None:
        # print(f"Input: {message}. History: {history} Subject: {subject}")
        docs = retriever.invoke(message)
        # print("docs", docs)
        knowledge = ""
        for doc in docs:
            knowledge += doc.page_content+"\n\n"
        
        partial_message = ""
        rag_prompt = f"""
        You are an assistent which answers questions based on knowledge which is provided to you.
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
    else:
        yield "There is no data in this subject yet. Please upload the file."

# initiate the Gradio app
with gr.Blocks() as chatbot:
    gr.Markdown("# A Personal Learning System")
    state = gr.State({
            "subject": str,
            "questions": [],
            "currentIdx": 0,
            "incorrect": []
        })
    
    with gr.Row():
        upload_file = gr.File(label="Upload PDF")
        with gr.Column():
            subject_name  = gr.Dropdown(['Add New'] + subject_list, label="Subject", info="Select the subject or create new one")
            new_subject  = gr.Textbox(label="New Subject", placeholder="Type the Subject", lines=2)

    with gr.Row():       
        upload_button = gr.Button("Upload")
        add_button = gr.Button("Add New")

    with gr.Column(visible=False) as system_block:
        system_area = gr.Textbox(label="Question", lines=1, interactive=False)

    with gr.Row():
        chat_button = gr.Button("Ask Question")
        quiz_button = gr.Button("Take Quiz")
        review_button = gr.Button("I want to review")
        
    with gr.Column(visible=False) as question_block:
        gr.Markdown("## Take the Quiz")
        output_area = gr.Textbox(label="Question", lines=1, interactive=False)
        ans_area = gr.Radio(["A", "B", "C", "D"], label="Your Answer")
        submit_button = gr.Button("Submit Answer")

    with gr.Column(visible=False) as chat_block:
        gr.Markdown("## Chat with the LLM")
        gr.ChatInterface(
            streamResponse,
            type="messages",
            additional_inputs=[subject_name]
        )

    add_button.click(addSubject, inputs=[subject_name, new_subject], outputs=[system_area, system_block, subject_name])
    upload_button.click(uploadFile, inputs=[upload_file, subject_name], outputs=[system_area, system_block])
    chat_button.click(startChatbot, inputs=[], outputs=[chat_block, question_block])
    quiz_button.click(startQuiz, inputs=[subject_name, state], outputs=[output_area, state, question_block, chat_block])
    review_button.click(reviewQuiz, inputs=[subject_name, state], outputs=[output_area, state, question_block, chat_block])
    submit_button.click(submitAns, inputs=[ans_area, state], outputs=[output_area, state])

# launch the Gradio app
chatbot.launch(pwa=True)