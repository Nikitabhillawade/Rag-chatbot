import streamlit as st
import requests

# Local development port pointer linking to our active FastAPI backend service
API_URL = "https://rag-chatbot-backend-0x3h.onrender.com"

st.set_page_config(page_title="Document Chatbot", layout="centered")
st.title("Chat With Your Documents")
st.caption("Upload a PDF or text file, then ask questions about it.")

# Instantiate persistent context array to preserve message history across reactive renders
if "messages" not in st.session_state:
    st.session_state.messages = []

# Structural UI Navigation Sidebar for processing local asset ingestions
with st.sidebar:
    st.subheader("Upload a document")
    file = st.file_uploader("Choose a file", type=["pdf", "txt", "md"])
    
    if file is not None and st.button("Upload & Index"):
        with st.spinner("Processing document..."):
            files = {"file": (file.name, file.getvalue())}
            resp = requests.post(f"{API_URL}/upload", files=files)
            
        if resp.status_code == 200:
            data = resp.json()
            st.success(f"Indexed {data.get('chunks_created', 0)} chunks from {file.name}")
        else:
            st.error("Upload failed")

# Render historical conversation logs sequentially on layout refresh
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Reactive prompt interaction framework capturing downstream runtime evaluations
question = st.chat_input("Ask a question about your uploaded document...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            resp = requests.post(f"{API_URL}/chat", json={"question": question})
            
        if resp.status_code == 200:
            data = resp.json()
            st.write(data["answer"])
            
            # Conditionally render target document structural search metadata citations
            if data.get("sources"):
                st.caption(f"Sources: {', '.join(data['sources'])}")
                
            st.session_state.messages.append({"role": "assistant", "content": data["answer"]})
        else:
            st.error("Something went wrong")