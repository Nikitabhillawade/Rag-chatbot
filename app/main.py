from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.ingest import ingest_document
from app.retrieve import retrieve_relevant_chunks
from app.llm import ask_llm
from app.schemas import ChatRequest, ChatResponse

app = FastAPI(title="Document RAG Chatbot")

# Enable global routing permissions for incoming frontend client connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    """Verifies that the API operational instance is healthy and reachable."""
    return {"status": "ok", "message": "Document RAG Chatbot API"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Accepts PDF, TXT, or MD streams, parses their textual content, and updates the vector database."""
    allowed = (".pdf", ".txt", ".md")
    if not file.filename.lower().endswith(allowed):
        raise HTTPException(400, "Please upload a PDF, TXT, or MD file")
   
    contents = await file.read()
    result = ingest_document(contents, file.filename)
    return result

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Executes a semantic vector database search, feeds the results to the LLM, and passes back the answer."""
    chunks = retrieve_relevant_chunks(request.question)

    if not chunks:
        return ChatResponse(
            answer="No documents have been uploaded yet — please upload one first.",
            sources=[]
        )

    # Run context assembly and dispatch the request to Llama 3.1
    answer = ask_llm(request.question, chunks)
    sources = list({c["source"] for c in chunks})

    return ChatResponse(answer=answer, sources=sources)