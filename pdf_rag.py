# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 02:30:39 2025

@author: zaher alkaei
"""
# ------------------------
# Imprting the libraries
# ------------------------

import fitz
import sys
import time
import threading
import subprocess
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.neighbors import NearestNeighbors

# ------------------------
# Extract text from PDF
# ------------------------
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        return "\n".join(page.get_text() for page in doc)
    except Exception as e:
        raise RuntimeError(f"Failed to load or read PDF: {e}")

# ------------------------
# Split text into chunks
# ------------------------
def split_text_into_chunks(text, chunk_size=800, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

# ------------------------
# Embed chunks
# ------------------------
def embed_text_chunks(chunks, model_name="all-MiniLM-L6-v2"):
    try:
        embedder = SentenceTransformer(model_name)
        embeddings = embedder.encode(chunks)
        return embeddings, embedder
    except Exception as e:
        raise RuntimeError(f"Embedding failed: {e}")

# ------------------------
# Build index
# ------------------------
def create_nn_index(embeddings):
    index = NearestNeighbors(n_neighbors=5, metric="cosine")
    index.fit(embeddings)
    return index

# ------------------------
# Retrieve top-k relevant chunks
# ------------------------
def retrieve_top_chunks(query, index, embedder, texts, k=5):
    q_vec = embedder.encode([query])
    distances, indices = index.kneighbors(q_vec, n_neighbors=k)
    return "\n".join([texts[i] for i in indices[0]])

# ------------------------
# Spinner class to show loading animation
# ------------------------
class Spinner:
    def __init__(self, message="Thinking"):
        self.message = message
        self.stop_running = False

    def start(self):
        def animate():
            while not self.stop_running:
                for dot_count in range(1, 4):
                    sys.stdout.write(f"\r{self.message}{'.' * dot_count}   ")
                    sys.stdout.flush()
                    time.sleep(0.5)
            sys.stdout.write("\r" + " " * (len(self.message) + 4) + "\r")
        self.thread = threading.Thread(target=animate)
        self.thread.start()

    def stop(self):
        self.stop_running = True
        self.thread.join()

# ------------------------
# Ask question using local LLM
# ------------------------
def ask_question_with_context(context, query, model="llama3.2:latest"):
    prompt = f"""You are a helpful assistant. Read the following text and answer the user's question clearly and concisely.

Context:
{context}

Question:
{query}
Answer:"""

    spinner = Spinner("Generating answer")
    try:
        spinner.start()
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode(),
            capture_output=True,
            timeout=120
        )
        spinner.stop()
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode() or "Unknown error running the model.")
        return result.stdout.decode()
    except subprocess.TimeoutExpired:
        spinner.stop()
        return "Error: Model response timed out. Try a shorter question or restart Ollama."
    except FileNotFoundError:
        spinner.stop()
        return "Error: Ollama not found. Make sure it’s installed and added to your PATH."
    except Exception as e:
        spinner.stop()
        return f"Unexpected error: {str(e)}"

# ------------------------
# Full RAG pipeline
# ------------------------
def run_rag_pipeline(pdf_path, question, index=None, embedder=None, chunks=None):
    try:
        if index is None or embedder is None or chunks is None:
            raw_text = extract_text_from_pdf(pdf_path)
            print(f"Extracted {len(raw_text)} characters from the PDF.")
            chunks = split_text_into_chunks(raw_text)
            print(f"Split into {len(chunks)} chunks.")
            embeddings, embedder = embed_text_chunks(chunks)
            index = create_nn_index(np.array(embeddings))
        context = retrieve_top_chunks(question, index, embedder, chunks)
        answer = ask_question_with_context(context, question)
        return answer, index, embedder, chunks
    except Exception as e:
        return f"An error occurred in the pipeline: {e}", index, embedder, chunks

# ------------------------
# Interactive CLI
# ------------------------
if __name__ == "__main__":
    print("📘 PDF Q&A Assistant is ready.")
    print("📝 Type a question to ask about your PDF. Type 'exit' to quit.\n")

    pdf_path = "book.pdf"
    index = embedder = chunks = None

    while True:
        try:
            question = input("❓ Your question: ")
            if question.lower() in ["exit", "quit"]:
                print("👋 Goodbye.")
                break
            answer, index, embedder, chunks = run_rag_pipeline(pdf_path, question, index, embedder, chunks)
            print("\n✅ Answer:\n", answer.strip(), "\n")
        except KeyboardInterrupt:
            print("\nInterrupted by user. Exiting.")
            break
        except Exception as e:
            print(f"Fatal error: {str(e)}")
