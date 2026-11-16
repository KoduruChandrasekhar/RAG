import os

# Every path is anchored to the project root rather than the working directory,
# so the scripts can be run from anywhere ("python src/evaluate.py" and
# "cd src && python evaluate.py" both work).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
FILINGS_DIR = os.path.join(RAW_DIR, "sec-edgar-filings")
PARSED_DIR = os.path.join(DATA_DIR, "parsed")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")

SRC_DIR = os.path.join(PROJECT_ROOT, "src")
TEST_QUESTIONS = os.path.join(SRC_DIR, "test_questions.json")

# Kept as-is deliberately: this is the on-disk ChromaDB collection key.
# Renaming it would orphan an existing store and force a full re-embed.
COLLECTION_NAME = "finrag_chunks"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GEMINI_MODEL = "gemini-3.6-flash"

TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"]

# Display names. Internally the pipeline works in tickers and raw section keys;
# anything a reader sees should be spelled out instead.
COMPANY_NAMES = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet",
    "NVDA": "NVIDIA",
    "AMZN": "Amazon",
}

SECTION_LABELS = {
    "Item_1A_Risk_Factors": "Risk Factors",
    "Item_7_MDA": "Management's Discussion & Analysis",
}

SECTION_LABELS_SHORT = {
    "Item_1A_Risk_Factors": "Risk Factors",
    "Item_7_MDA": "MD&A",
}


def company_name(ticker):
    return COMPANY_NAMES.get(ticker, ticker)


def section_label(section, short=False):
    table = SECTION_LABELS_SHORT if short else SECTION_LABELS
    return table.get(section, section)


def filing_year(filing_id):
    """Pull the filing year out of an SEC accession number.

    Accession numbers look like 0000320193-24-000123, where the middle pair of
    digits is the year the filing was submitted.
    """
    parts = str(filing_id).split("-")
    if len(parts) >= 2 and len(parts[1]) == 2 and parts[1].isdigit():
        return 2000 + int(parts[1])
    return None


# Used to scope a question to the company it names.
TICKER_ALIASES = {
    "AAPL": ["apple", "iphone", "ipad", "mac", "aapl"],
    "MSFT": ["microsoft", "azure", "windows", "msft"],
    "GOOGL": ["google", "alphabet", "youtube", "googl", "goog"],
    "NVDA": ["nvidia", "nvda", "geforce", "cuda"],
    "AMZN": ["amazon", "aws", "amzn"],
}
