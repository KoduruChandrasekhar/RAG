import os
from groq import __name
from sec_edgar_downloader import Downloader
from dotenv import load_dotenv 

load_dotenv()

def download_filings():
    raw_dir =os.path.join("data","raw")
    os.makedirs(raw_dir,exist_ok=True)

    dl = Downloader("rag-project","chandrasekhar14437@gmail.com",raw_dir)
    tickers = ["AAPL","MSFT","GOOGL","NVDA","AMZN"]
    for ticker in tickers:
        print(f"Downloading 10-k filings for{ticker}...")
        dl.get("10-K",ticker,limit=3)
        print(f"Done : {ticker}")

    print("\nAll downloads complete.Check data/raw/sec-edgar-filings")
if( __name__ == "__main__"):
    download_filings()
        