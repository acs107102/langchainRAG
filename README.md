# 📚 AI Learning Platform (Powered by LangChain & RAG)

This is an AI-powered learning platform that utilizes **LangChain** and **Retrieval-Augmented Generation (RAG)** techniques to process PDF course materials, provide intelligent Q&A, generate quizzes, and support error review for better learning retention.

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

## 🚀 Getting Started

### 1️⃣ Install Dependencies

Ensure you have **Python 3.8+** installed, then run:

```bash
pip install -r requirements.txt
```

### 2️⃣ Generate Database

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

### 3️⃣ Start the System

```bash
python main.py
```

---

## 📌 To-Do / Possible Improvements

- ✅ **Error review functionality**  
- ✅ **Manual PDF upload support**  
- ⏳ **Support for multiple LLM backends (e.g., GPT-4, Llama-2)**  
- ⏳ **Web-based user interface**  
- ⏳ **More flexible quiz modes**  

---

## 📜 License

This project is licensed under the **MIT License**. Feel free to fork and improve it.
