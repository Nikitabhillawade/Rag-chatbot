# 🤖 AI-Powered Document RAG Chatbot

An intelligent, full-stack Retrieval-Augmented Generation (RAG) web application that allows users to upload documents (PDFs, TXT, MD) and have context-aware conversations about their contents. 

This project uses an optimized single-server architecture deployed completely in the cloud, leveraging local vector embeddings and lightning-fast inference servers.

🚀 **[Click Here to Try the Live Demo!](https://rag-chatbot-4vk93qonjetycgct76cn7k.streamlit.app)**

---

## ✨ Features

* **Multi-Format Ingestion:** Seamless parsing for `.pdf`, `.txt`, and `.md` files via smart data loaders.
* **Semantic Text Splitting:** Dynamically slices document layouts into precise text chunks with controlled overlap to preserve contextual continuity.
* **On-the-Fly Embeddings:** Instantiates a transformer-based embedding model directly inside the system container to index semantics.
* **Vector Database Management:** Leverages an in-memory storage manager to perform sub-second similarity matching based on mathematical distance.
* **Ultra-Fast LLM Inference:** Connects directly to Groq's high-efficiency LPUs running Llama 3.1 for instantaneous responses.
* **Source Citations:** Transparently labels the precise document source chunk from which the answer was retrieved.

---

## 🛠️ Tech Stack & Architecture

* **Frontend UI:** [Streamlit](https://streamlit.io/) (Interactive Dashboard)
* **Orchestration Framework:** [LangChain](https://www.langchain.com/) (Document Loaders, Text Splitters, Vector Wrappers)
* **Vector Database:** [ChromaDB](https://www.trychroma.com/) (In-memory structural semantic indexer)
* **Embedding Model:** `all-MiniLM-L6-v2` (Hugging Face Sentence Transformers)
* **LLM Provider:** [Groq Cloud API](https://groq.com/) (Running `llama-3.1-8b-instant`)

---

## 🚀 Local Installation & Setup

If you want to pull this code repository down and run it locally on your machine, follow these steps:

### 1. Clone the Repository
```bash
git clone [https://github.com/Nikitabhillawade/Rag-chatbot.git](https://github.com/Nikitabhillawade/Rag-chatbot.git)
cd Rag-chatbot