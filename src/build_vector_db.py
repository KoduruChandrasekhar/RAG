import os
import json
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv

load_dotenv()


def build_vector_db():
    parsed_dir = os.path.join("data", "parsed")
    embeddings_dir = os.path.join("data", "embeddings")

    # Check parsed data exists
    json_files = [f for f in os.listdir(parsed_dir) if f.endswith(".json")]
    if not json_files:
        print("ERROR: data/parsed/ is empty. Run parse_documents.py first.")
        return

    print(f"Found {len(json_files)} JSON files to process.")

    # Initialize ChromaDB — stores data on disk so it persists
    client = chromadb.PersistentClient(path=embeddings_dir)

    # This embedding model runs LOCALLY — no API cost, no internet needed
    # First run downloads it (~130MB), after that it's cached
    print("Loading embedding model (downloads ~130MB on first run)...")
    ef = SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-small-en-v1.5"
    )

    # Delete existing collection if rebuilding from scratch
    try:
        client.delete_collection("finrag_chunks")
        print("Deleted existing collection (rebuilding fresh).")
    except Exception:
        pass

    # Create collection with cosine similarity
    # Cosine similarity measures the angle between vectors — best for text
    collection = client.create_collection(
        name="finrag_chunks",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )

    # Load all chunks from every JSON file
    all_texts = []
    all_ids = []
    all_metadatas = []

    for filename in json_files:
        filepath = os.path.join(parsed_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        for chunk in data["chunks"]:
            all_texts.append(chunk["text"])
            all_ids.append(chunk["chunk_id"])
            all_metadatas.append({
                "ticker": chunk["ticker"],
                "filing_id": chunk["filing_id"],
                "section": chunk["section"]
            })

    print(f"Total chunks to embed: {len(all_texts)}")
    print("Embedding chunks (this takes a few minutes)...")

    # Add in batches of 100 to avoid memory issues
    batch_size = 100
    total_batches = (len(all_texts) - 1) // batch_size + 1

    for i in range(0, len(all_texts), batch_size):
        end = min(i + batch_size, len(all_texts))
        collection.add(
            documents=all_texts[i:end],
            ids=all_ids[i:end],
            metadatas=all_metadatas[i:end]
        )
        current_batch = i // batch_size + 1
        print(f"  Batch {current_batch}/{total_batches} done")

    print(f"\nVector database built successfully.")
    print(f"Total chunks stored: {collection.count()}")
    print(f"Location: data/embeddings/")


if __name__ == "__main__":
    build_vector_db()