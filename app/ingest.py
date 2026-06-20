from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import io
import uuid

# Load a local, lightweight open-source embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize a persistent local vector database directory
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="documents")

def extract_text(file_bytes: bytes, filename: str) -> str:
    """Extracts raw string text from either a PDF or plain text file."""
    if filename.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        return file_bytes.decode("utf-8", errors="ignore")

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """Splits text into uniform chunks with an overlap boundary so context isn't lost."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def ingest_document(file_bytes: bytes, filename: str) -> dict:
    """Rips text, creates chunks, builds vector embeddings, and saves to the DB."""
    text = extract_text(file_bytes, filename)
    if not text.strip():
        return {"chunks_created": 0, "error": "No extractable text found"}

    chunks = chunk_text(text)
    # Convert text strings into floating-point math lists (embeddings)
    embeddings = embedder.encode(chunks).tolist()

    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]

    # Commit securely to our Chroma Vector Store
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas
    )

    return {"chunks_created": len(chunks), "filename": filename}