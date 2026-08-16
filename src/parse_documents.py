import os
import json
import re
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()


def clean_html(raw_content):
    
    soup = BeautifulSoup(raw_content, "lxml")
    text = soup.get_text(separator=" ")
    
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_sections(text, ticker, filing_id):
    sections = {}

    item_1a_matches = []

    # Strategy 1: Standard regex patterns
    item_1a_patterns = [
        r'(?i)item\s+1a[\.\s\-\:]+risk\s+factors',
        r'(?i)item\s+1a[—\.\s]+risk\s+factors',
        r'(?i)(?:^|\n)\s*1a\.\s+risk\s+factors'
    ]
    for pattern in item_1a_patterns:
        item_1a_matches = list(re.finditer(pattern, text))
        if item_1a_matches:
            break

    # Strategy 2: Specific fallback for MSFT layout structure
    if not item_1a_matches and ticker == "MSFT":
        # Microsoft 10-Ks explicitly contain uppercase "ITEM 1A. RISK FACTORS" or standalone headers
        item_1a_matches = list(re.finditer(r'(?i)ITEM\s+1A\.\s+RISK\s+FACTORS', text))
        if not item_1a_matches:
            item_1a_matches = list(re.finditer(r'(?i)RISK\s+FACTORS', text))

    item_1b_matches = list(re.finditer(r'(?i)item\s+1b[\.\s\-\:]+unresolved', text))
    item_2_matches = list(re.finditer(r'(?i)item\s+2[\.\s\-\:]+propert', text))

    if item_1a_matches:
        # For MSFT, if multiple 'RISK FACTORS' match, pick the one safely past the 15% mark of the document to avoid ToC
        if ticker == "MSFT" and len(item_1a_matches) > 1:
            valid_matches = [m for m in item_1a_matches if m.start() > len(text) * 0.15]
            target_match = valid_matches[0] if valid_matches else item_1a_matches[-1]
            start = target_match.end()
        else:
            start = item_1a_matches[-1].end()

        end = start + 120000 if ticker == "MSFT" else start + 50000  
        
        for match in item_1b_matches + item_2_matches:
            if match.start() > start:
                end = match.start()
                break  
                
        sections["Item_1A_Risk_Factors"] = text[start:end].strip()
    else:
        print(f"  WARNING: Item 1A not found — {ticker} {filing_id}")

    # 2. Item 7 (MD&A) Extraction
    item_7_patterns = [
        r'(?i)item\s+7[\.\s\-\:]+management[’\']?s\s+discussion',
        r'(?i)item\s+7[—\.\s]+management[’\']?s\s+discussion'
    ]
    
    item_7_matches = []
    for pattern in item_7_patterns:
    # Safely search for MD&A
        item_7_matches = list(re.finditer(pattern, text))
        if item_7_matches:
            break

    if not item_7_matches and ticker == "MSFT":
        item_7_matches = list(re.finditer(r'(?i)MANAGEMENT\u2019S\s+DISCUSSION\s+AND\s+ANALYSIS', text))

    item_7a_matches = list(re.finditer(r'(?i)item\s+7a[\.\s\-\:]+quantitative', text))
    item_8_matches = list(re.finditer(r'(?i)item\s+8[\.\s\-\:]+financial\s+statements', text))

    if item_7_matches:
        start = item_7_matches[-1].end()
        end = start + 100000  
        
        for match in item_7a_matches + item_8_matches:
            if match.start() > start:
                end = match.start()
                break
                
        sections["Item_7_MDA"] = text[start:end].strip()
    else:
        print(f"  WARNING: Item 7 not found — {ticker} {filing_id}")

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
    raw_dir = os.path.join("data", "raw", "sec-edgar-filings")
    parsed_dir = os.path.join("data", "parsed")
    os.makedirs(parsed_dir, exist_ok=True)

    if not os.path.exists(raw_dir):
        print(f"ERROR: {raw_dir} not found. Run ingest_sec_data.py first.")
        return

    total_files = 0
    total_chunks = 0

    for ticker in os.listdir(raw_dir):
        ticker_path = os.path.join(raw_dir, ticker, "10-K")

        if not os.path.isdir(ticker_path):
            continue

        for filing_folder in os.listdir(ticker_path):
            filing_path = os.path.join(
                ticker_path, filing_folder, "full-submission.txt"
            )

            if not os.path.exists(filing_path):
                print(f"  SKIP: No full-submission.txt for {ticker} {filing_folder}")
                continue

            print(f"\nProcessing: {ticker} — {filing_folder}")

            
            with open(filing_path, "r", encoding="utf-8", errors="ignore") as f:
                raw_content = f.read()

            
            clean_text = clean_html(raw_content)

            
            sections = extract_sections(clean_text, ticker, filing_folder)

            if not sections:
                print(f"  SKIPPING: No sections extracted")
                continue

            
            output = {
                "ticker": ticker,
                "filing_id": filing_folder,
                "chunks": []
            }

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

            
            output_path = os.path.join(
                parsed_dir, f"{ticker}_{filing_folder}.json"
            )
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2)

            total_files += 1

    print(f"\nDone.")
    print(f"Files processed: {total_files}")
    print(f"Total chunks created: {total_chunks}")
    print(f"Output: data/parsed/")


if __name__ == "__main__":
    process_all_filings()