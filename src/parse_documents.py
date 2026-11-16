import os
import json
import re
from bs4 import BeautifulSoup
from dotenv import load_dotenv

import paths

load_dotenv()

def extract_10k_document(full_submission):
    pattern = re.compile(
        r'<TYPE>10-K\b.*?<TEXT>(.*?)</TEXT>',
        re.DOTALL | re.IGNORECASE
    )
    match = pattern.search(full_submission)
    if match:
        return match.group(1)
    return full_submission

def clean_html(raw_content):
    content = extract_10k_document(raw_content)
    soup = BeautifulSoup(content, "lxml")
    
    for tag in soup(["script", "style"]):
        tag.decompose()
        
    text = soup.get_text(separator=" ")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def spaced(word):
    # HTML styling sometimes splits a heading word apart ("RIS K FACTORS" in
    # Microsoft's filings), so allow optional whitespace between the letters.
    return r'\s*'.join(re.escape(c) for c in word)


SEPARATOR = r'[\.\-\:\s–—]+'

MDA_TITLE = (
    r"management[’\'s\s]*discussion\s+and\s+analysis"
    r"(\s+of\s+financial\s+condition\s+and\s+results\s+of\s+operations)?"
)

# In the table of contents a heading is followed by a page number and the next
# item: "Risk Factors 16 Item 1B. Unresolved Staff Comments 30 ...".
TOC_PATTERN = re.compile(r'\d{1,3}\s+Item\s+\d', re.IGNORECASE)

# A cross-reference continues the surrounding sentence rather than opening a
# section: it is followed by punctuation, a dash, or a phrase such as
# "of this Form 10-K" or "for a discussion of".
CROSSREF_LEAD = ('”', '“', '"', '’', ',', ';', '–', '—', ')')
CROSSREF_DOT_QUOTE = re.compile('^\\s*\\.\\s*[”“"\']')
CROSSREF_PHRASE = re.compile(
    r'(?i)^\W{0,3}(of|in|and|to|for|under|see|is|are|our|this)\b.{0,80}?'
    r'(this\s+(form|annual|report)|part\s+i|10-k|for\s+a\s+discussion|'
    r'for\s+additional|note\s+\d|included\s+in)'
)


def is_table_of_contents(text, position):
    return bool(TOC_PATTERN.search(text[position:position + 250]))


def is_cross_reference(text, position):
    window = text[position:position + 200]
    stripped = window.lstrip()
    if stripped[:1] in CROSSREF_LEAD:
        return True
    if CROSSREF_DOT_QUOTE.match(window):
        return True
    return bool(CROSSREF_PHRASE.match(stripped))


def find_best_match(text, patterns, after_position=0):
    # "Item 1A. Risk Factors" occurs many times in a 10-K as a cross-reference
    # and once in the table of contents, so return the earliest occurrence that
    # is neither -- that is the real heading.
    all_matches = []
    for pattern in patterns:
        try:
            matches = list(re.finditer(pattern, text))
            valid = [m for m in matches if m.start() >= after_position]
            all_matches.extend(valid)
        except re.error:
            continue

    if not all_matches:
        return None

    all_matches.sort(key=lambda m: m.start())

    for match in all_matches:
        if is_table_of_contents(text, match.end()):
            continue
        if is_cross_reference(text, match.end()):
            continue
        return match

    return all_matches[0]


def extract_sections(text, ticker, filing_id):
    sections = {}
    min_start = int(len(text) * 0.08)

    item_1a_patterns = [
        r'(?i)item\s+1a' + SEPARATOR + spaced('risk') + r'\s+' + spaced('factors'),
        r'(?m)^\s*RISK\s+FACTORS\s*$',
    ]

    item_1a_end_patterns = [
        r'(?i)item\s+1b' + SEPARATOR + spaced('unresolved'),
        r'(?i)item\s+1c' + SEPARATOR + spaced('cybersecurity'),
        r'(?i)item\s+2' + SEPARATOR + r'propert',
    ]

    start_match = find_best_match(text, item_1a_patterns, min_start)
    if start_match:
        start = start_match.end()
        end = min(start + 120000, len(text))

        end_match = find_best_match(text, item_1a_end_patterns, start + 500)
        if end_match and end_match.start() < end:
            end = end_match.start()

        content = text[start:end].strip()
        if len(content.split()) > 200:
            sections["Item_1A_Risk_Factors"] = content

    item_7_patterns = [
        r'(?i)item\s+7' + SEPARATOR + MDA_TITLE,
        r'(?i)' + MDA_TITLE,
    ]

    item_7_end_patterns = [
        r'(?i)item\s+7a' + SEPARATOR + r'quantitative',
        r'(?i)item\s+8' + SEPARATOR + spaced('financial') + r'\s+' + spaced('statements'),
    ]

    start_match = find_best_match(text, item_7_patterns, min_start)
    if start_match:
        start = start_match.end()
        end = min(start + 200000, len(text))

        end_match = find_best_match(text, item_7_end_patterns, start + 500)
        if end_match and end_match.start() < end:
            end = end_match.start()

        content = text[start:end].strip()
        if len(content.split()) > 200:
            sections["Item_7_MDA"] = content

    return sections


def chunk_text(text, chunk_size=150, overlap=50):
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = ' '.join(words[i:i + chunk_size])
        if len(chunk.strip()) > 50:
            chunks.append(chunk)
    return chunks

def process_all_filings():
    raw_dir = paths.FILINGS_DIR
    parsed_dir = paths.PARSED_DIR
    os.makedirs(parsed_dir, exist_ok=True)

    if not os.path.exists(raw_dir):
        print(f"ERROR: {raw_dir} not found. Run ingest_sec_data.py first.")
        return

    old_files = [f for f in os.listdir(parsed_dir) if f.endswith(".json")]
    for f in old_files:
        os.remove(os.path.join(parsed_dir, f))

    total_files = 0
    total_chunks = 0

    for ticker in sorted(os.listdir(raw_dir)):
        ticker_path = os.path.join(raw_dir, ticker, "10-K")
        if not os.path.isdir(ticker_path):
            continue

        for filing_folder in sorted(os.listdir(ticker_path)):
            filing_path = os.path.join(ticker_path, filing_folder, "full-submission.txt")
            if not os.path.exists(filing_path):
                continue

            print(f"Processing: {ticker} — {filing_folder}")
            with open(filing_path, "r", encoding="utf-8", errors="ignore") as f:
                raw_content = f.read()

            clean_text = clean_html(raw_content)
            sections = extract_sections(clean_text, ticker, filing_folder)

            if not sections:
                continue

            output = {"ticker": ticker, "filing_id": filing_folder, "chunks": []}
            for section_name, content in sections.items():
                chunks = chunk_text(content)
                print(f"  {section_name}: {len(chunks)} chunks")
                for idx, chunk in enumerate(chunks):
                    output["chunks"].append({
                        "chunk_id": f"{ticker}_{filing_folder}_{section_name}_{idx}",
                        "ticker": ticker,
                        "filing_id": filing_folder,
                        "section": section_name,
                        "text": chunk
                    })
                total_chunks += len(chunks)

            output_path = os.path.join(parsed_dir, f"{ticker}_{filing_folder}.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2)
            total_files += 1

    print(f"\nDone. Files processed: {total_files} | Total chunks: {total_chunks}")

if __name__ == "__main__":
    process_all_filings()