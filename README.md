# AI Resume Q&A Bot using LangChain (RAG)

## 📌 Overview
The **AI Resume Q&A Bot** is a command-line application that allows users to ask natural language questions about a **PDF resume** and receive **accurate, context-aware answers**.

The project is built using **LangChain** and follows the **Retrieval Augmented Generation (RAG)** pattern to ensure responses are generated **strictly from resume content**.

---

## 🎯 Problem Statement
Traditional LLM-based systems cannot directly process PDF documents and often produce unreliable or hallucinated answers when queried without proper grounding.

This project addresses these challenges by:
- Extracting structured text from a resume PDF
- Converting the content into semantic vector embeddings
- Retrieving only the most relevant sections of the resume
- Injecting retrieved context into a controlled prompt

This approach improves **accuracy, relevance, and consistency**.

---

## 🧠 Solution Approach (RAG Pipeline)
The system implements **Retrieval Augmented Generation (RAG)** to ensure all answers are grounded in the resume.

### Processing Flow
1. Resume PDF is loaded and converted into text  
2. Text is split into semantically meaningful chunks  
3. Chunks are transformed into vector embeddings  
4. Embeddings are stored in a FAISS vector store  
5. User queries are embedded and matched against stored vectors  
6. Relevant resume chunks are retrieved  
7. Retrieved context is injected into a prompt template  
8. The LLM generates a response based only on this context  

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
LLM (via Ollama)
↓
Answer

---

## 🛠️ Tech Stack
- **Python**
- **LangChain**
- **Ollama**
- **FAISS**
- **PyPDF**

---

## ✨ Key Features
- Natural language question answering over resume content
- Context-grounded responses using RAG
- Deterministic extraction for basic information (name, email, phone)
- Reduced hallucination through strict prompt constraints
- Command-line interface for simplicity and clarity
