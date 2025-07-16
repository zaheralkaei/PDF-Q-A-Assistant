# PDF-Q-A-Assistant
This is a simple Retrieval-Augmented Generation (RAG) system that allows you to ask questions about the content of any PDF file. The script uses local embeddings and a local language model (via [Ollama](https://ollama.com)) to retrieve relevant context from the PDF and generate an answer.

---

## 🔍 What this code does

1. **Extracts** text from a PDF using `PyMuPDF`
2. **Splits** the text into overlapping chunks using `LangChain`
3. **Embeds** the chunks using `SentenceTransformers`
4. **Indexes** them with `scikit-learn` for fast retrieval (cosine similarity)
5. **Queries** a local LLM using [Ollama](https://ollama.com) to generate answers based on retrieved context

---

## ⚙️ Requirements

### 1. Python packages

Install dependencies using:

```bash
pip install -r requirements.txt
```

Your `requirements.txt` should include:

```
numpy==1.26.4
PyMuPDF==1.23.22
sentence-transformers==2.6.1
langchain==0.2.1
scikit-learn==1.7.0
```

---

### 2. Install Ollama (for local LLM)

Ollama lets you run models like LLaMA and Mistral locally.

- Download and install from: [https://ollama.com/download](https://ollama.com/download)

Once installed, pull a model (e.g., LLaMA 3):

```bash
ollama run llama3
```

This will download the model if it’s not already installed.  
Make sure the model name matches what’s used in the script (`llama3.2:latest`) or adjust accordingly.

---

## 📄 Usage

1. Place your PDF file in the project directory and name it `book.pdf`,  
   or change the file path in `pdf_rag.py`.

2. Run the script:

```bash
python pdf_rag.py
```

3. Ask your questions interactively in the terminal:

```text
Your question: What is the main argument in chapter 2?
```

Type `exit` or `quit` to leave the session.

---

## 📁 File structure

```
.
├── pdf_rag.py             # Main script
├── book.pdf               # Your input PDF file
├── requirements.txt       # Dependencies
└── README.md              # This file
```

---

## 🧠 Example use case

You have a long academic paper or a policy document in PDF format and want quick answers without reading the whole file. This script helps you extract relevant content and ask questions using a local AI model — no API keys needed.

---

## 📝 Notes

- Works well for small to medium PDFs. For very large documents, consider adding disk-based caching.
- Embeddings and index are kept in memory during runtime.
- You can swap out the model name (e.g., `llama3.2`) with any supported Ollama model.

---

## 📄 License

MIT
