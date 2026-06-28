import os
import chromadb
from src import config
from src import llm

class VectorStore:
    def __init__(self):
        # Initialize PersistentClient using the configured DB path
        self.client = chromadb.PersistentClient(path=config.DB_DIR)
        # Create or retrieve collection
        self.collection = self.client.get_or_create_collection(name="graphmind_documents")

    def add_chunks(self, chunks: list[dict]):
        """
        Embeds and adds a list of chunks to the vector store.
        Each chunk is expected to be: {"text": chunk_text, "source": filename, "page": page_number}
        """
        if not chunks:
            return
            
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [{"source": chunk["source"], "page": chunk["page"]} for chunk in chunks]
        
        # Generate unique IDs based on source, page, and chunk index
        ids = [f"{chunk['source']}_p{chunk['page']}_c{idx}" for idx, chunk in enumerate(chunks)]
        
        # Generate embeddings via our llm module
        embeddings = llm.get_embeddings(texts)
        
        # Add to collection
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Queries ChromaDB and returns top_k matching chunks with documents, metadata, and scores.
        """
        query_embedding = llm.get_embeddings([query])[0]
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        retrieved_chunks = []
        if results and results["documents"] and results["documents"][0]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            ids = results["ids"][0]
            distances = results["distances"][0] if "distances" in results and results["distances"] else [0.0] * len(docs)
            
            for doc, meta, cid, dist in zip(docs, metas, ids, distances):
                retrieved_chunks.append({
                    "text": doc,
                    "source": meta.get("source", "unknown"),
                    "page": meta.get("page", 0),
                    "id": cid,
                    "distance": dist
                })
        return retrieved_chunks

    def reset(self):
        """Clears the collection to re-ingest documents."""
        try:
            self.client.delete_collection("graphmind_documents")
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(name="graphmind_documents")
