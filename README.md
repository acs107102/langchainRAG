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

## 📌 To-Do / Possible Improvements

- ✅ **Error review functionality**  
- ✅ **Manual PDF upload support**  
- ✅ **Ability to add new subjects dynamically**  
- ✅ **Support for PDF and web-based document loading**  
- ⏳ **Support for multiple LLM backends (e.g., GPT-4, Llama-2)**  
- ⏳ **Web-based user interface**  
- ⏳ **More flexible quiz modes**  
- ⏳ **Enhanced mistake tracking and review features**  

---

## 📜 License

This project is licensed under the **MIT License**. Feel free to fork and improve it.
