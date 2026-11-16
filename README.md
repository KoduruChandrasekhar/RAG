# 🔎 FilingScope

![tests](https://github.com/KoduruChandrasekhar/RAG/actions/workflows/tests.yml/badge.svg)

Hybrid retrieval over SEC 10-K filings. Ask a question in plain English and get an answer grounded in — and cited back to — the actual filings of Apple, Amazon, Google, Microsoft and NVIDIA.

Built as an end-to-end RAG pipeline: ingest → parse → embed → retrieve → generate, with evaluation at both the retrieval and answer layers.

---

## 🚀 Pipeline

| Stage | Script | What it does |
|---|---|---|
| 1. Ingestion | `src/ingest_sec_data.py` | Downloads the last 3 10-K filings per ticker from SEC EDGAR |
| 2. Parsing | `src/parse_documents.py` | Strips HTML, locates **Item 1A (Risk Factors)** and **Item 7 (MD&A)**, chunks at 150 words with 50-word overlap |
| 3. Indexing | `src/build_vector_db.py` | Embeds every passage locally and persists them in ChromaDB (cosine) |
| 4. Querying | `src/query_engine.py` | Scoping → hybrid retrieval → re-ranking → cited answer |
| 5. Evaluation | `src/evaluate_retrieval.py` | Retrieval metrics, **no LLM calls, no API quota** |
| | `src/evaluate.py` | RAGAS answer-quality metrics (needs quota) |
| UI | `src/app.py` | Streamlit front end, including an architecture walkthrough |

---

## 🧠 How it works

```
BUILD ONCE   SEC EDGAR ──▶ parse + chunk ──▶ embed (local) ──▶ ChromaDB
                           Item 1A + 7       bge-small-en        cosine
                           150w / 50 ovlp

PER QUESTION  question ──▶ ┌─ dense  (ChromaDB, top 20) ─┐
              scoped to   │                             ├─▶ RRF ──▶ cross-encoder ──▶ Gemini
              a company   └─ sparse (BM25,     top 20) ─┘   k=60      top 5          cited answer
```

### 1. Company scoping

If a question names a company — *Apple*, *Azure*, *AWS*, *CUDA* — retrieval is restricted to that company's filings. Without this, a question about one company readily surfaces another's text, because risk language is near-identical across big tech. If the filter matches nothing, the search falls back to the whole corpus rather than answering with no context.

### 2. Two retrievers, because they fail in opposite directions

- **Dense** (ChromaDB) matches on *meaning* — finds "supply chain disruption" for a question about "manufacturing problems", but can miss an exact term it has no strong embedding for.
- **Sparse** (BM25) matches on *words* — intended to catch exact identifiers like an export-control classification `3A090.a`, where embeddings blur, but it misses paraphrases entirely.

> On the current 40-question set BM25 does **not** measurably improve results over dense retrieval alone — see the [ablation](#ablation--does-each-stage-earn-its-place). It is kept because the questions that would justify it (exact identifiers, product codes) are under-represented in that set, not because the benefit has been demonstrated.

### 3. Reciprocal Rank Fusion

Each passage scores `1 / (60 + rank)` summed across the lists it appears in. A passage both retrievers rank highly beats one only a single retriever liked — **agreement is the signal**. Using ranks rather than raw scores is what makes it safe to combine a cosine similarity with a BM25 score, which are not on comparable scales.

### 4. Cross-encoder re-ranking

Fusion is cheap and approximate. The cross-encoder reads the question and passage *together* instead of comparing two independent vectors — far more accurate, but too slow for 2,700 passages. Running it only on the fused shortlist buys the accuracy at a workable cost. Top 5 survive.

### 5. Duplicate suppression and company balance

Annual reports restate the same boilerplate year after year, so the top results
filled up with one paragraph repeated from three filings. Passages whose word
overlap exceeds 0.70 against an already-selected passage are dropped — a
threshold calibrated against the corpus, where genuine restatements score 0.75+
and unrelated passages sit near 0.18.

Comparative questions get a second correction. Asking how Microsoft and NVIDIA
each describe export controls returned **five NVIDIA passages and no
Microsoft**: a pooled search handed NVIDIA 31 of 36 candidates and re-ranking
removed Microsoft entirely. When several companies are in scope the search now
runs **once per company**, so each brings its own shortlist, and slots are
reserved per company. That question now returns 3 NVIDIA and 2 Microsoft.

### 6. Grounded generation

The 5 passages, each tagged with ticker, filing ID and section, become the context. Gemini is instructed to use only that context and cite it as `[Source N]`, so every claim traces back to a filing.

---

## 📐 Section extraction (the part that decides answer quality)

`Item 1A. Risk Factors` appears many times in a 10-K: once in the table of contents, and repeatedly as cross-references like *"see Item 1A. Risk Factors of this Form 10-K"*. Matching naively lands in the wrong part of the document and silently files MD&A text under risk factors — which poisons retrieval without any visible error.

`find_best_match` returns the **earliest match that is neither**:

- a **table-of-contents entry** — followed by a page number and the next item
- a **cross-reference** — followed by punctuation, a dash, or a phrase like *"of this Form 10-K"*

Headings are also matched letter-by-letter with optional gaps, because HTML styling splits words apart — Microsoft's filings render the heading as `ITEM 1A. RIS K FACTORS`.

The **Corpus** tab in the UI shows per-filing passage counts for both sections. A zero there means extraction failed for that filing.

---

## 🛠️ Tech Stack

* **Language:** Python 3.13
* **Parsing:** Beautiful Soup, lxml, Regex
* **Vector Database:** ChromaDB
* **Embeddings:** Sentence Transformers (`BAAI/bge-small-en-v1.5`, runs locally)
* **Re-ranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
* **Sparse retrieval:** `rank-bm25`
* **LLM:** Google Gemini API (`gemini-3.6-flash`)
* **Evaluation:** RAGAS + a custom retrieval evaluator
* **UI:** Streamlit
* **Config:** `python-dotenv`

---

## ⚙️ Setup

### 1. Clone

```bash
git clone https://github.com/KoduruChandrasekhar/RAG.git
```

### 2. Install

```bash
pip install -r requirements.txt
```

### 3. Configure

Copy `.env.example` to `.env` and fill it in:

```
GEMINI_API_KEY = your_key_here
SEC_CONTACT_EMAIL = you@example.com
```

`GEMINI_API_KEY` comes from [Google AI Studio](https://aistudio.google.com/apikey). `SEC_CONTACT_EMAIL` is required by SEC EDGAR, which rejects bulk requests without a contact address in the User-Agent.

---

## ▶️ Running

Build the index once, in order. Each script resolves its own paths, so it runs from any directory.

```bash
python src/ingest_sec_data.py
```

```bash
python src/parse_documents.py
```

```bash
python src/build_vector_db.py
```

Stage 3 downloads the embedding model (~130 MB) on first run, then caches it. Expect a few minutes to embed the corpus.

### Web UI

```bash
streamlit run src/app.py
```

Three tabs: **Ask**, **How it works** (architecture diagram and a walkthrough of every stage), and **Corpus** (what is indexed, per filing). Models load on the first question, not at page load.

### Command line

```bash
python src/query_engine.py "What manufacturing risks did Apple identify related to China?"
```

| Flag | Effect |
|---|---|
| `-t, --ticker SYM` | Restrict to a ticker, repeatable (`-t MSFT -t NVDA`) |
| `-s, --section {risk,mda}` | Restrict to Item 1A or Item 7 |
| `-k, --top-k N` | Passages to keep after re-ranking (default 5) |
| `-a, --all-companies` | Search everything, skipping company detection |
| `-r, --retrieval-only` | Show passages without calling the LLM — **no quota used** |
| `-i, --interactive` | Keep asking without reloading the models |

```bash
python src/query_engine.py -i -r
```

Interactive mode pays the ~20s model load once instead of per question.

---

## 📊 Evaluation

### Retrieval — free, run it as often as you like

```bash
python src/evaluate_retrieval.py
```

Every question in `src/test_questions.json` carries `expected_tickers` and `expected_sections`. A retrieved passage counts as relevant when it comes from an expected ticker *and* section. No LLM is involved.

| Metric | Meaning | Current |
|---|---|---|
| Hit Rate | Questions with at least one relevant passage | 1.000 |
| MRR | 1 / rank of the first relevant passage | 0.915 |
| Precision@5 | Share of retrieved passages that are relevant | 0.840 |
| Ticker Accuracy | Questions where company detection was exactly right | 1.000 |

Measured over **40 labelled questions** covering both sections for all five
companies, including five comparative questions naming two companies at once.
An earlier 11-question set pinned Hit Rate and MRR at 1.000, which made them
useless for telling configurations apart — the set was grown specifically to
fix that.

### Ablation — does each stage earn its place?

```bash
python src/ablation.py
```

Turns each stage off in turn and re-scores the same 40 questions. No LLM, so it
can be re-run after any change to check it actually helped.

| configuration | Hit | MRR | P@5 |
|---|---|---|---|
| Dense only (no BM25, no re-rank) | 1.000 | **0.958** | **0.870** |
| Sparse only (BM25, no re-rank) | 0.950 | 0.793 | 0.750 |
| Hybrid, no re-ranker | 1.000 | 0.899 | **0.870** |
| Hybrid + re-ranker | 1.000 | 0.910 | 0.845 |
| Full pipeline | 1.000 | 0.915 | 0.840 |
| Full, company scoping **off** | 1.000 | 0.830 | 0.695 |

At 40 questions one passage changing place moves P@5 by 0.005, so differences
above ~0.02 are real. Read honestly, this says:

- **Company scoping is the one stage that clearly pays.** Turning it off costs
  **0.145** — by far the largest effect measured, and the reason a question
  about one company no longer returns another's filings.
- **BM25 does not pay for itself here.** Sparse-only is the weakest row by a
  wide margin, and hybrid fusion matches plain dense retrieval on P@5 (0.870
  both) while scoring *worse* on MRR. BM25 exists to catch exact identifiers
  like an export-control classification `3A090.a`; this question set probably
  under-represents that case rather than disproving it.
- **The re-ranker costs precision and buys a little ranking quality**
  (P@5 −0.025, MRR +0.011). It is not currently justified on this evidence.
- **Deduplication is neutral on these metrics** (P@5 −0.005, inside noise;
  MRR +0.005) — but it removes every near-duplicate from the results, which
  these metrics do not measure at all.

**The honest conclusion is that the metric is too coarse to settle this.**
Relevance here means "expected company *and* expected section", which many
passages satisfy equally. The stages that score badly — re-ranking and
deduplication — are exactly the ones optimising for things the metric ignores:
topical fit within a section, and not repeating yourself. A passage-level
relevance labelling would be needed to judge them properly.

Nothing has been removed from the pipeline on the strength of these numbers.
What they do establish is that **company scoping earns its place outright**, and
that the re-ranker's contribution is unproven.

---

### Passage-level relevance labels

The metrics above score a passage relevant when it comes from an expected
company and an expected section. That is a *category* test — many passages
satisfy it equally, so it cannot tell a good passage from a great one, and it
systematically penalises the two stages that optimise for what it cannot see:
re-ranking (topical fit *within* a section) and deduplication.

Judging passages directly fixes that, and unlocks **nDCG** — the metric designed
to say whether a ranker puts the best material first.

```bash
python src/build_label_pool.py
```

This pools what **every** ablation configuration retrieves, not just the current
pipeline — otherwise a rival configuration gets marked down for returning
passages nobody looked at. It writes `label_passages.html`: open it in a
browser and judge with the number keys.

| key | grade | meaning |
|---|---|---|
| `2` | relevant | directly answers part of the question |
| `1` | partial | related, or useful background |
| `0` | irrelevant | does not help |

`←` goes back to correct a judgement. Progress is kept in the browser, so you
can stop and resume. When done, use the Download button and save the file over
`src/relevance_labels.json`.

The current pool is **535 passages across 40 questions** — roughly an hour of
work. 505 of them are *contested*: returned by some configurations but not
others, so they are what decides which configuration wins. Those are presented
first, so a partial pass is still worth doing.

**Partial labelling works.** `evaluate_retrieval.py` and `ablation.py` pick the
labels up automatically, use a human judgement wherever one exists, fall back to
the category test everywhere else, and print how much of the result rests on
real labels:

```
Passage-level labels: 35 judged across 3/40 questions.
Unlabelled passages fall back to the company+section test.

configuration                           Hit    MRR    P@5  nDCG@5
```

Labels are keyed by question text and chunk id, so re-running retrieval or
changing the pipeline does not invalidate them.

---

### Answer quality — needs quota

```bash
python src/evaluate.py --limit 2
```

Faithfulness, Answer Relevancy, Context Precision and Context Recall via RAGAS.

| metric | result | note |
|---|---|---|
| **Faithfulness** | **0.938** | measured over 2 questions |
| Answer Relevancy | not yet measured | see below |
| Context Precision | not yet measured | quota exhausted mid-run |
| Context Recall | not yet measured | quota exhausted mid-run |

Faithfulness is the metric that matters most here: it asks whether the answer's
claims are actually supported by the passages retrieved. **0.938** says the
answers stay grounded in the filings rather than drifting into the model's own
knowledge — which is the whole point of the citation prompt.

Treat it as a spot check, not a benchmark: two questions is far too few, and the
other three metrics have never completed a run.

**One failure here was not about quota.** Answer Relevancy defaults to
`strictness=3`, asking the model for three candidate generations at once;
`gemini-3.6-flash` refuses with `400 INVALID_ARGUMENT: Multiple candidates is
not enabled for this model`, so the metric could never score against this model
however much quota was available. It is now pinned to `strictness=1`, which also
costs a third of the requests.

| Flag | Effect |
|---|---|
| `-n, --limit N` | Evaluate only the first N questions |
| `-m, --metrics ...` | Run a subset of the metrics |
| `-k, --top-k N` | Passages retrieved per question |

Before running, it prints an estimate of how many Gemini requests the run needs and warns when that exceeds the free-tier daily allowance — so you find out before the run dies halfway through rather than after.

> **Free-tier quota.** `gemini-3.6-flash` allows 20 `generate_content` requests per day on the free tier. A full RAGAS run over several questions and four metrics needs considerably more, so expect `N/A` scores from quota exhaustion unless you are on a paid plan or cut the question set down. This is exactly why `evaluate_retrieval.py` exists. When the LLM is unavailable the pipeline degrades gracefully: generation reports that it failed and the retrieved sources are still returned.

---

## 🧪 Tests

```bash
python -m pytest tests/ -q
```

163 tests covering section-boundary detection (contents entries, cross-references, split headings), chunk overlap arithmetic, RRF fusion, ticker and section filters, re-ranking, retry behaviour for both API and network failures, preflight checks, quota budgeting, duplicate suppression, per-company balancing, word-boundary company detection, and the retrieval metrics. No network calls, no API key and no vector store needed — the engine is built with `__new__` and handed stubs, which is also what lets the suite run in CI.

The suite reports ~90s, but the tests themselves run in well under a second; the rest is importing `torch` and `transformers`. For quick iteration on parsing:

```bash
python -m pytest tests/test_parse_documents.py -q
```

That runs in about a second.

---

## 🧷 Robustness

The pipeline is built to survive the conditions it actually runs in.

**Transient failures are retried.** The Gemini free tier returns `503` on load
spikes and `429` on rate limits, and connections drop. All three are retried
with exponential backoff; anything else (a `400`, a bad key) is raised
immediately rather than silently swallowed. Both failure families are covered by
tests — the `httpx.ConnectError` case was added after a real evaluation run died
on it.

**Failure degrades, it does not crash.** When generation cannot complete, the
answer field says so plainly and **the retrieved sources are still returned**, so
a quota-exhausted session is still useful for reading filings.

**Missing build steps are named.** Running a query before the index exists prints
which script to run instead of an opaque ChromaDB traceback:

```
No parsed filings in .../data/parsed.
Run:  python src/ingest_sec_data.py
      python src/parse_documents.py
      python src/build_vector_db.py
```

**Dependencies are pinned.** RAGAS especially — it renamed its dataset columns in
0.2 and changed its result type, and the old names are accepted *without error*
while mapping to nothing. An unpinned upgrade reports every metric as `N/A` and
tells you nothing is wrong.

---

## 🧰 Task runner

A `Makefile` wraps the common commands:

```bash
make build        # ingest + parse + index
make eval         # retrieval metrics, no quota
make ablate       # what each retrieval stage contributes
make test-fast    # parsing tests, ~1s
make ui           # launch Streamlit
```

`make help` lists them all. On Windows without `make`, run the underlying
`python` commands shown above.

---

## 🌐 Deploying a live demo

The app is a long-running Streamlit server holding two transformer models in
memory, so it needs a host that can run a persistent Python process. **Vercel
cannot run it** — serverless functions are stateless, and the dependencies alone
are ~766 MB against a ~250 MB bundle limit.

**Hugging Face Spaces** fits, and needs no GitHub:

```bash
pip install huggingface_hub
huggingface-cli login
python deploy/push_to_space.py --space <your-username>/filingscope --dry-run
python deploy/push_to_space.py --space <your-username>/filingscope
```

It uploads ~45 MB: the app, the engine, the parsed sections and the ChromaDB
store. The 236 MB of raw filings stay local — they are only needed to rebuild
the index. `src/` is kept as a subfolder because `paths.py` derives the project
root from its own location.

**Answer generation is optional.** Without a `GEMINI_API_KEY` secret the Space
still searches and shows passages; only the written answer is skipped, and the
UI says so. That is usually the right default for a public demo, since the
Gemini free tier allows 20 generations per day in total — shared by every
visitor.

---

## 📁 Layout

```
rag-project/
├── data/
│   ├── raw/         # downloaded filings        (gitignored)
│   ├── parsed/      # sections + chunks         (gitignored)
│   └── embeddings/  # ChromaDB store            (gitignored)
├── src/
│   ├── paths.py              # project paths, model names, ticker aliases
│   ├── ingest_sec_data.py
│   ├── parse_documents.py
│   ├── build_vector_db.py
│   ├── query_engine.py       # engine + CLI
│   ├── evaluate_retrieval.py # quota-free metrics
│   ├── ablation.py           # what each stage contributes
│   ├── relevance.py          # passage labels + nDCG
│   ├── build_label_pool.py   # builds the labelling tool
│   ├── evaluate.py           # RAGAS metrics
│   ├── app.py                # Streamlit UI
│   └── test_questions.json
├── tests/
├── deploy/                   # Hugging Face Space packaging
├── .github/workflows/        # CI: tests on 3.11 / 3.12 / 3.13
├── Makefile
├── requirements.txt
├── .env.example
└── .env                      # not committed
```

Everything under `data/` is regenerable from the three build scripts, which is why it is gitignored.
