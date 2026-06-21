import streamlit as st
import os
import tempfile
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

st.set_page_config(page_title="Document Chatbot", layout="centered")
st.title("Chat With Your Documents")
st.caption("Upload a PDF or text file, then ask questions about it.")

# Securely extract API keys from standard cloud runtime environments
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("Please add your GROQ_API_KEY to Streamlit Secrets to proceed.")
    st.stop()

# Instantiate global Groq client link
client = Groq(api_key=GROQ_API_KEY)

# Cache embedding resource models to optimize cross-render execution times
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embedding_model = load_embeddings()

# Initialize empty state trackers across layout execution refreshes
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Structural UI Sidebar Navigation framework for processing document drops
with st.sidebar:
    st.subheader("Upload a document")
    file = st.file_uploader("Choose a file", type=["pdf", "txt", "md"])
    
    if file is not None and st.button("Upload & Index"):
        with st.spinner("Processing document..."):
            # Stream volatile payload bits into local ephemeral file storage wrappers
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.name}") as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                if file.name.endswith(".pdf"):
                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()
                else:
                    loader = TextLoader(tmp_path, encoding="utf-8")
                    docs = loader.load()
                
                # Distribute text layers into discrete semantic chunk arrays
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_documents(docs)
                
                for chunk in chunks:
                    chunk.metadata["source"] = file.name
                
                # Assemble in-memory Chroma instance tracking the newly minted chunks
                st.session_state.vector_store = Chroma.from_documents(
                    documents=chunks,
                    embedding=embedding_model
                )
                st.success(f"Indexed {len(chunks)} chunks from {file.name}")
            except Exception as e:
                st.error(f"Error processing file: {e}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

# Render historical conversation streams
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Process downstream user message queries
question = st.chat_input("Ask a question about your uploaded document...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            context = ""
            sources = []
            
            # Query vector structural index databases if document context is active
            if st.session_state.vector_store is not None:
                docs = st.session_state.vector_store.similarity_search(question, k=3)
                context = "\n\n".join([doc.page_content for doc in docs])
                sources = list(set([doc.metadata.get("source", "Unknown") for doc in docs]))
            
            system_prompt = (
                "You are a helpful assistant. Answer the user's question using only the provided context below.\n"
                f"Context:\n{context}"
            )
            
            try:
                # Dispatch execution payloads directly to Groq LPU server layers
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question}
                    ],
                    model="llama-3.1-8b-instant"
                )
                answer = chat_completion.choices[0].message.content
                st.write(answer)
                
                if sources:
                    st.caption(f"Sources: {', '.join(sources)}")
                    
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"API Error: {e}")