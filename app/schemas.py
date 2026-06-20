from pydantic import BaseModel

class ChatRequest(BaseModel):
    """Enforces that the incoming user payload contains a valid question string."""
    question: str

class ChatResponse(BaseModel):
    """Structures the API response payload with the generated answer string and data source origins."""
    answer: str
    sources: list