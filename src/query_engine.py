import os
import json
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# Configure Gemini API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


class FinRAGEngine:
    def __init__(self):
        print("Loading FinRAG engine...")
        embeddings_dir = os.path.join("data", "embeddings")
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=embeddings_dir)
        
        # Load local embedding model
        self.ef = SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-small-en-v1.5"
        )
        
        # Get existing collection
        self.collection = self.client.get_collection(
            name="finrag_chunks",
            embedding_function=self.ef
        )
        
        # Initialize Gemini model using a current supported endpoint
        self.gemini = genai.GenerativeModel("gemini-3.5-flash")
        print(f"Engine ready. Chunks in database: {self.collection.count()}")

    def retrieve(self, query, n_results=5):
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        chunks = []
        for i in range(len(results["documents"][0])):
            chunks.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "id": results["ids"][0][i]
            })
        return chunks

    def _generate_answer(self, query, chunks):
        context_str = ""
        for idx, chunk in enumerate(chunks):
            meta = chunk["metadata"]
            context_str += f"\n--- Source [{idx+1}]: Ticker: {meta['ticker']} | Filing: {meta['filing_id']} | Section: {meta['section']} ---\n"
            context_str += chunk["text"] + "\n"

        prompt = f"""
You are an expert financial analyst. Answer the user's question accurately using only the provided context extracted from SEC 10-K filings. Cite your sources using the format [Source X] where appropriate.

Context:
{context_str}

Question: {query}

Answer:
"""
        response = self.gemini.generate_content(prompt)
        return response.text

    def query(self, question):
        print(f"\nSearching database for: '{question}'")
        retrieved_chunks = self.retrieve(question, n_results=5)
        
        print(f"Retrieved {len(retrieved_chunks)} relevant chunks. Generating answer...")
        answer = self._generate_answer(question, retrieved_chunks)
        return answer


if __name__ == "__main__":
    engine = FinRAGEngine()
    question = "What are Apple's main risks related to China?"
    result = engine.query(question)
    print("\n" + "="*40 + "\nANSWER:\n" + "="*40)
    print(result)