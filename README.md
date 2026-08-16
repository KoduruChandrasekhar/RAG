# Financial RAG Project 📈🤖

An end-to-end Retrieval-Augmented Generation (RAG) pipeline designed to ingest, parse, chunk, embed, and query SEC 10-K filings from major technology companies (Apple, Amazon, Google, Microsoft, and NVIDIA) using local vector search and Google's Gemini API.

---

## 🚀 Project Architecture

1. **Data Ingestion (`src/ingest_sec_data.py`)**: Downloads raw 10-K filings from SEC EDGAR.
2. **Document Parsing (`src/parse_documents.py`)**: Cleans HTML, extracts specific sections (Item 1A: Risk Factors, Item 7: MD&A), and chunks text with overlap.
3. **Vector Database (`src/build_vector_db.py`)**: Computes local embeddings using Sentence Transformers (`BAAI/bge-small-en-v1.5`) and stores them persistently in **ChromaDB**.
4. **Query Engine (`src/query_engine.py`)**: Performs semantic similarity searches and generates cited financial analysis via the Gemini API.

---

## 🛠️ Tech Stack

* **Language:** Python
* **Parsing:** Beautiful Soup, Regex
* **Vector Database:** ChromaDB
* **Embeddings:** Hugging Face Sentence Transformers (`bge-small-en-v1.5`)
* **LLM:** Google Gemini API (`gemini-3.5-flash`)
* **Environment Management:** `python-dotenv`

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/KoduruChandrasekhar/RAG.git](https://github.com/KoduruChandrasekhar/RAG.git)
