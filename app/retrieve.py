from app.ingest import embedder, collection

def retrieve_relevant_chunks(query: str, top_k: int = 4) -> list:
    """Converts a user query into an embedding and fetches the top_k most semantically similar chunks."""
    # Convert the raw question into the same mathematical vector space as the documents
    query_embedding = embedder.encode([query]).tolist()

    # Query ChromaDB for the closest contextual matches
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    documents = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []

    # Map the text chunks to their original document filenames for reference tagging
    return [
        {"text": doc, "source": meta.get("source", "unknown")}
        for doc, meta in zip(documents, metadatas)
    ]