# AI Resume Q&A Bot using LangChain (RAG, Free Local LLM)

## 📌 Overview
The AI Resume Q&A Bot is a command-line based application that allows users to ask natural language questions about a **PDF resume** and receive **accurate, context-aware answers**.

The project is built using **LangChain** and follows the **Retrieval Augmented Generation (RAG)** approach.  
To keep the project completely free and accessible, it uses a **locally hosted Large Language Model (LLM) via Ollama**, eliminating the need for paid APIs such as OpenAI.

---

## 🎯 Problem Statement
Large Language Models (LLMs) cannot directly read PDF files and often hallucinate answers when asked questions without proper context.

This project addresses these challenges by:
- Extracting text from a resume PDF
- Storing the content as semantic vector embeddings
- Retrieving only the most relevant resume sections
- Injecting retrieved context into the LLM using structured prompt engineering

This ensures **high accuracy, relevance, and reliability**.

---

## 🧠 Solution Approach (RAG Pipeline)
The system uses **Retrieval Augmented Generation (RAG)** to generate answers strictly from the resume content.

### Flow:
1. Resume PDF is loaded and converted to text
2. Text is split into smaller semantic chunks
3. Chunks are converted into vector embeddings
4. Embeddings are stored in a FAISS vector database
5. User queries are embedded and matched semantically
6. Relevant resume chunks are retrieved
7. Retrieved context is injected into a prompt template
8. Local LLM generates the final answer

---

## 🏗️ Architecture

Resume PDF
↓
Document Loader
↓
Text Chunking
↓
Vector Embeddings
↓
FAISS Vector Store
↓
Retriever
↓
Prompt Template
↓
Local LLM (Ollama)
↓
Answer


---

## 🛠️ Tech Stack
- Python
- LangChain
- Ollama (Local LLM)
- FAISS (Vector Database)
- PyPDF

---

## ✨ Key Features
- Ask questions about a resume using natural language
- Accurate answers using RAG
- Reduced hallucination by restricting LLM context
- Fully free (no paid APIs)
- Clean CLI-based interface
- Resume and interview ready project

---

## ⚙️ Setup & Installation

### 1. Install Ollama
Download from https://ollama.com and install.

Pull a free model:
```bash
ollama pull mistral
