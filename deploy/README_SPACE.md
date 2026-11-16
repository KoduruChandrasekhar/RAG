---
title: FilingScope
emoji: 🔎
colorFrom: indigo
colorTo: gray
sdk: streamlit
sdk_version: 1.41.1
app_file: src/app.py
pinned: false
license: mit
short_description: Hybrid retrieval over SEC 10-K filings from five tech companies
---

# 🔎 FilingScope

Ask questions about SEC 10-K filings from Apple, Amazon, Alphabet, Microsoft and
NVIDIA, and get answers cited back to the filings themselves.

Two sections of each filing are indexed — **Item 1A (Risk Factors)** and
**Item 7 (Management's Discussion & Analysis)** — across the three most recent
annual reports per company.

## How it works

Open the **How it works** tab in the app for the full walkthrough. In short:

1. **Company scoping** — a question that names a company is answered only from
   that company's filings.
2. **Two searches in parallel** — meaning-based (vector) and keyword (BM25),
   because they miss different things.
3. **Reciprocal Rank Fusion** — passages both searches agree on rise to the top.
4. **Cross-encoder re-ranking** — a slower, more accurate model reads the
   question against each shortlisted passage.
5. **Grounded generation** — only the retrieved passages are given to the model,
   with instructions to cite them and invent nothing.

Every answer comes with the passages behind it, and a **How this question was
analysed** panel showing what the pipeline actually did.

## Answer generation

If this Space has no `GEMINI_API_KEY` secret configured, search works normally
and the matching passages are shown — only the written answer is skipped.

Source: the project README in the repository this Space was built from.
