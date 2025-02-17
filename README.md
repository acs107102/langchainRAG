# 📚 Learning Platform

This is an AI-powered learning platform that uses **LangChain** and **Retrieval-Augmented Generation (RAG)** techniques to process PDF course materials, provide Q&A, generate quizzes, and support error review for better learning retention.

## ✨ Features

- **📂 Course Material Management**
  - Uses `generateDB.py` to automatically categorize PDFs into different subjects based on local course folders.
  - Allows manual PDF uploads within the system and dynamic subject creation.

- **🤖 AI-Powered Q&A & Quiz Generation**
  - Enables users to ask subject-specific questions, with responses generated from relevant documents.
  - Automatically generates quizzes to aid learning and revision.

- **📌 Error Review System**
  - Stores incorrectly answered questions for targeted revision and practice.

---

## 🛠️ Tech Stack

- **Framework:** [LangChain](https://www.langchain.com/)
- **Vector Database:** ChromaDB  
- **LLM Backend:** OpenAI API
- **Frontend:** Gradio

---

## 🔗 Deployment

You can try the demo deployed on Hugging Face Spaces:

👉 [Live Demo on Hugging Face](https://huggingface.co/spaces/acs107102/Learning_Platform_RAG)

> ⚠ **Note:** The Hugging Face deployment does not retain uploaded files permanently. Every time you access the platform, you will need to re-upload your course materials.

---

## 🚀 Getting Started

### 1️. Install Dependencies

Ensure you have **Python 3.8+** installed, then run:

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

Create a `.env` file in the project root and add your OpenAI API key:

```
OPENAI_API_KEY=your-api-key-here
```


### 3. Generate Database

Your course materials should be structured as follows:

```
School/
  ├── Subject1/
  │   ├── file1.pdf
  │   ├── file2.pdf
  │   └── ...
  ├── Subject2/
  │   ├── file3.pdf
  │   ├── file4.pdf
  │   └── ...
  └── ...
```

To process the course materials and create the database, run:

```bash
python generateDB.py
```

### 4. Start the System

```bash
python main.py
```

---

## 📸 Screenshot

Q&A

![image-1](image/image-1.png)

Quiz

![image-2](image/image-2.png)

---

## 📌 To-Do / Possible Improvements

- ✅ **Error review functionality**  
- ✅ **Manual PDF upload support**  
- ✅ **Ability to add new subjects dynamically**  
- ✅ **Support for PDF and web-based document loading**  
- ⏳ **Support for multiple LLM backends (e.g. Llama-2)**  
- ⏳ **Web-based user interface**  
- ⏳ **More quiz modes (e.g. Short-answer questions)**  
- ⏳ **Enhanced mistake tracking and review features**  
