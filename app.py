import re

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


# =========================
# BASIC INFO EXTRACTION
# =========================

def extract_name(documents):
    first_page_lines = documents[0].page_content.split("\n")
    for line in first_page_lines[:5]:
        clean = line.strip()
        if 2 <= len(clean.split()) <= 4 and clean.replace(" ", "").isalpha():
            return clean
    return None


def extract_email(documents):
    text = " ".join(doc.page_content for doc in documents)
    match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    return match.group(0) if match else None


def extract_phone(documents):
    text = " ".join(doc.page_content for doc in documents)
    match = re.search(r"(\+?\d{1,3}[\s-]?)?\d{10}", text)
    return match.group(0) if match else None


# =========================
# OUTPUT SANITIZATION
# =========================

def clean_answer(text):
    """
    Prevents run-on generations and unrelated examples.
    """
    stop_markers = [
        "John Doe",
        "The following information",
        "Rules:",
        "Question:",
        "Proof by contradiction",
    ]

    for marker in stop_markers:
        if marker in text:
            text = text.split(marker)[0]

    return text.strip()


# =========================
# LOAD RESUME
# =========================

loader = PyPDFLoader("sample_resume.pdf")
documents = loader.load()

name = extract_name(documents)
email = extract_email(documents)
phone = extract_phone(documents)


# =========================
# RAG PIPELINE
# =========================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.from_documents(chunks, embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


prompt_template = """
You are a strict resume analyzer.

Answer ONLY using the resume context.
Do NOT infer, assume, or guess.
Do NOT use external knowledge.

If information is not explicitly mentioned, say:
"Not mentioned in the resume."

Resume Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

llm = Ollama(
    model="phi",
    temperature=0,
    num_ctx=512,
    num_predict=150,
    stop=["\n\n", "Question:", "Answer:"]
)


qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt}
)


# =========================
# CLI LOOP
# =========================

print("\nAI Resume Q&A Bot (type 'exit' to quit)\n")

while True:
    query = input("Ask a question about the resume:\n> ").strip()
    if query.lower() == "exit":
        break

    q = query.lower()

    # ---- BASIC INFO (NO LLM) ----
    if "name" in q:
        print(f"\nAnswer:\n {name if name else 'Not mentioned in the resume.'}\n")
        continue

    if "email" in q:
        print(f"\nAnswer:\n {email if email else 'Not mentioned in the resume.'}\n")
        continue

    if "phone" in q or "contact number" in q:
        print(f"\nAnswer:\n {phone if phone else 'Not mentioned in the resume.'}\n")
        continue

    if "contact details" in q:
        details = []
        if email:
            details.append(f"Email: {email}")
        if phone:
            details.append(f"Phone: {phone}")

        if details:
            print("\nAnswer:\n " + ", ".join(details) + "\n")
        else:
            print("\nAnswer:\n Not mentioned in the resume.\n")
        continue

    # ---- SUBJECTIVE BLOCK ----
    subjective_keywords = ["strongest", "best", "expert", "most experienced"]

    if any(word in q for word in subjective_keywords):
        print("\nAnswer:\n The resume does not explicitly mention this.\n")
        continue

    # ---- RAG FOR EVERYTHING ELSE ----
    result = qa_chain.invoke({"query": query})
    final_answer = clean_answer(result["result"])
    print("\nAnswer:\n", final_answer, "\n")
