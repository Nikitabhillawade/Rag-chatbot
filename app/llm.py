import os
from groq import Groq
from dotenv import load_dotenv

# Hydrate our environment with the hidden GROQ_API_KEY from the .env file
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def build_prompt(question: str, chunks: list) -> str:
    """Assembles the system instructions, context chunks, and user query into a clean prompt."""
    context = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in chunks
    )
    
    # Anti-hallucination instruction block to ground the LLM's logic
    return f"""You are a helpful assistant that answers questions using ONLY the provided context. 
If the answer isn't in the context, say you don't know based on the uploaded documents — do not make anything up. 

Context:
{context} 

Question: {question} 

Answer:"""

def ask_llm(question: str, chunks: list) -> str:
    """Dispatches the grounded context prompt to the Llama 3.1 model hosting engine."""
    prompt = build_prompt(question, chunks)
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,   # Kept low to keep answers highly focused and factual
        max_tokens=600,
    )
    
    return response.choices[0].message.content