"""Streamlit front end for FilingScope.

Run with:  streamlit run src/app.py
"""
import collections
import glob
import json
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paths
from query_engine import SECTION_CHOICES, FilingScopeEngine, PipelineNotBuilt

st.set_page_config(page_title="FilingScope", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Loading models and building the BM25 index...")
def load_engine():
    # Cached across reruns -- loading the two transformer models takes ~20s.
    return FilingScopeEngine()


@st.cache_data
def corpus_stats():
    rows = []
    sections = collections.Counter()
    for path in sorted(glob.glob(os.path.join(paths.PARSED_DIR, "*.json"))):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        per_section = collections.Counter(c["section"] for c in data["chunks"])
        sections.update(per_section)
        year = paths.filing_year(data["filing_id"])
        rows.append({
            "Company": paths.company_name(data["ticker"]),
            "Filed": year or "-",
            "Risk Factors": per_section["Item_1A_Risk_Factors"],
            "MD&A": per_section["Item_7_MDA"],
            "Total passages": len(data["chunks"]),
        })
    return rows, sections


ARCHITECTURE_SVG = """
<!doctype html>
<meta charset="utf-8">
<style>html,body{margin:0;padding:0;background:transparent}</style>
<svg viewBox="0 0 980 300" xmlns="http://www.w3.org/2000/svg"
     style="width:100%;height:auto;font-family:system-ui,sans-serif">
  <defs>
    <marker id="a" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0 L10 5 L0 10 z" fill="#8b93a7"/>
    </marker>
    <style>
      :root {
        --box-fill:#eef2fb; --box-line:#6b7bb5;
        --hot-fill:#fdeee8; --hot-line:#d98060;
        --title:#1f2733; --sub:#5d6678; --faint:#8b93a7;
      }
      @media (prefers-color-scheme: dark) {
        :root {
          --box-fill:#1d2437; --box-line:#7f8fc9;
          --hot-fill:#36241d; --hot-line:#e09272;
          --title:#e8ecf5; --sub:#a6b0c4; --faint:#7d869b;
        }
      }
      .box  { fill:var(--box-fill); stroke:var(--box-line); stroke-width:1.5; rx:8 }
      .hot  { fill:var(--hot-fill); stroke:var(--hot-line); stroke-width:1.5; rx:8 }
      .t    { font-size:13px; font-weight:600; fill:var(--title) }
      .s    { font-size:11px; fill:var(--sub) }
      .lbl  { font-size:11px; fill:var(--faint); letter-spacing:.06em }
      .ln   { stroke:var(--faint); stroke-width:1.5; marker-end:url(#a); fill:none }
    </style>
  </defs>

  <!-- offline pipeline -->
  <text x="10" y="20" class="lbl">BUILD ONCE (offline)</text>
  <rect class="box" x="10"  y="32" width="150" height="56"/>
  <text x="85" y="54" class="t" text-anchor="middle">SEC EDGAR</text>
  <text x="85" y="72" class="s" text-anchor="middle">15 x 10-K filings</text>

  <rect class="box" x="195" y="32" width="170" height="56"/>
  <text x="280" y="50" class="t" text-anchor="middle">Parse + chunk</text>
  <text x="280" y="66" class="s" text-anchor="middle">Item 1A + Item 7</text>
  <text x="280" y="80" class="s" text-anchor="middle">150 words / 50 overlap</text>

  <rect class="box" x="400" y="32" width="160" height="56"/>
  <text x="480" y="50" class="t" text-anchor="middle">Embed</text>
  <text x="480" y="66" class="s" text-anchor="middle">bge-small-en-v1.5</text>
  <text x="480" y="80" class="s" text-anchor="middle">runs locally</text>

  <rect class="box" x="595" y="32" width="150" height="56"/>
  <text x="670" y="54" class="t" text-anchor="middle">ChromaDB</text>
  <text x="670" y="72" class="s" text-anchor="middle">cosine, persisted</text>

  <line class="ln" x1="162" y1="60" x2="193" y2="60"/>
  <line class="ln" x1="367" y1="60" x2="398" y2="60"/>
  <line class="ln" x1="562" y1="60" x2="593" y2="60"/>

  <!-- query path -->
  <text x="10" y="135" class="lbl">EVERY QUESTION (online)</text>
  <rect class="hot" x="10"  y="147" width="150" height="50"/>
  <text x="85" y="168" class="t" text-anchor="middle">Your question</text>
  <text x="85" y="185" class="s" text-anchor="middle">scoped to a company</text>

  <rect class="box" x="195" y="128" width="170" height="40"/>
  <text x="280" y="145" class="t" text-anchor="middle">Dense search</text>
  <text x="280" y="160" class="s" text-anchor="middle">ChromaDB, top 20</text>

  <rect class="box" x="195" y="178" width="170" height="40"/>
  <text x="280" y="195" class="t" text-anchor="middle">Sparse search</text>
  <text x="280" y="210" class="s" text-anchor="middle">BM25, top 20</text>

  <rect class="box" x="400" y="150" width="130" height="46"/>
  <text x="465" y="170" class="t" text-anchor="middle">RRF fusion</text>
  <text x="465" y="186" class="s" text-anchor="middle">k = 60</text>

  <rect class="box" x="565" y="150" width="155" height="46"/>
  <text x="642" y="170" class="t" text-anchor="middle">Cross-encoder</text>
  <text x="642" y="186" class="s" text-anchor="middle">re-rank to top 5</text>

  <rect class="hot" x="755" y="150" width="155" height="46"/>
  <text x="832" y="170" class="t" text-anchor="middle">Gemini</text>
  <text x="832" y="186" class="s" text-anchor="middle">cited answer</text>

  <line class="ln" x1="162" y1="162" x2="193" y2="148"/>
  <line class="ln" x1="162" y1="182" x2="193" y2="198"/>
  <line class="ln" x1="367" y1="148" x2="398" y2="165"/>
  <line class="ln" x1="367" y1="198" x2="398" y2="181"/>
  <line class="ln" x1="532" y1="173" x2="563" y2="173"/>
  <line class="ln" x1="722" y1="173" x2="753" y2="173"/>

  <line class="ln" x1="670" y1="90" x2="670" y2="126" stroke-dasharray="4 3"/>
  <text x="680" y="112" class="s">index read at query time</text>

  <text x="755" y="215" class="s">optional - skip for</text>
  <text x="755" y="229" class="s">retrieval-only mode</text>
</svg>
"""

def render_analysis(question, result):
    """Explain what the system did with this particular question."""
    trace = result.get("trace")
    if not trace:
        return

    companies = [paths.company_name(t) for t in result["tickers"]]
    sections = [paths.section_label(x) for x in result["sections"]]

    if trace["scope_source"] == "detected":
        scope_line = (f"Recognised **{', '.join(companies)}** in your wording, "
                      f"so only that company's filings were searched.")
    elif trace["scope_source"] == "chosen":
        scope_line = f"You restricted the search to **{', '.join(companies)}**."
    elif trace["scope_source"] == "all":
        scope_line = "You asked for every company, so nothing was filtered out."
    else:
        scope_line = ("No company was named, so all five were searched. Naming "
                      "one usually gives a sharper answer.")

    section_line = (f"Limited to **{', '.join(sections)}**."
                    if sections else
                    "Both Risk Factors and MD&A were searched.")

    with st.expander("How this question was analysed"):
        st.markdown(f"**1 · Reading the question** — {scope_line} {section_line}")

        st.markdown(
            f"**2 · Two searches in parallel** — a meaning-based search returned "
            f"**{trace['dense_hits']}** passages and a keyword search returned "
            f"**{trace['sparse_hits']}**. They are run together because they miss "
            f"different things: meaning-based search handles paraphrases but can "
            f"skip an exact term, keyword search does the reverse.")

        st.markdown(
            f"**3 · Merging the two lists** — combined into "
            f"**{trace['fused_candidates']}** candidates. Passages that both "
            f"searches ranked highly rise to the top, because agreement between "
            f"two different methods is a strong signal.")

        st.markdown(
            f"**4 · Close reading** — a re-ranking model read your question "
            f"against each of **{trace['reranked_shortlist']}** candidates. "
            f"This is slow, so it only runs on the shortlist.")

        dropped = trace.get("duplicates_dropped", 0)
        if dropped:
            st.markdown(
                f"**5 · Removing repetition** — **{dropped}** of the best-scoring "
                f"passages repeated something already selected, so they were "
                f"replaced. Filings restate the same wording year after year, "
                f"and a repeat tells you nothing new.")
        else:
            st.markdown(
                f"**5 · Removing repetition** — the best **{trace['kept']}** "
                f"passages were all distinct, so none needed replacing.")

        if "answer" in result:
            st.markdown(
                "**6 · Writing the answer** — those passages, and nothing else, "
                "were given to the language model, with instructions to cite "
                "them and to invent nothing.")
        else:
            st.markdown(
                "**6 · Writing the answer** — skipped, because *Show passages "
                "only* is on.")

        if trace["widened_to_full_corpus"]:
            st.warning("Nothing matched those filters, so the search was widened "
                       "to the whole corpus. The passages above may be off-topic.")


st.title("🔎 FilingScope")
st.caption("Hybrid retrieval over SEC 10-K filings — Apple, Amazon, Google, "
           "Microsoft and NVIDIA.")

ask_tab, how_tab, corpus_tab = st.tabs(["Ask", "How it works", "Corpus"])

# ----------------------------------------------------------------------- Ask

with ask_tab:
    with st.sidebar:
        st.header("Search settings")

        company_choices = {paths.company_name(t): t for t in paths.TICKERS}
        chosen_companies = st.multiselect(
            "Companies", sorted(company_choices),
            help="Leave empty and the company is taken from your question.")
        tickers = [company_choices[c] for c in chosen_companies]

        section_choices = {paths.section_label(k): k for k in paths.SECTION_LABELS}
        chosen_sections = st.multiselect(
            "Parts of the filing", list(section_choices),
            help="Leave empty to search both.")

        top_k = st.slider("Passages to use", 1, 15, 5)

        can_generate = bool(os.getenv("GEMINI_API_KEY"))
        retrieval_only = st.checkbox(
            "Show passages only", value=not can_generate,
            disabled=not can_generate,
            help="Skip writing an answer and just show the matching passages. "
                 "Uses no API quota."
                 if can_generate else
                 "No API key is configured on this deployment, so only the "
                 "matching passages are shown.")

        st.divider()
        # Read the count from the parsed files rather than the engine, so that
        # opening the app (or the docs tabs) does not pay the ~20s model load.
        _, _sections = corpus_stats()
        st.metric("Passages searched", f"{sum(_sections.values()):,}")
        if can_generate:
            st.caption("Models load on your first question.")
        else:
            st.caption("Search is live. Answer writing is off on this "
                       "deployment — no API key configured.")

    question = st.text_input(
        "Question",
        placeholder="What manufacturing risks did Apple identify related to China?")

    if st.button("Ask", type="primary") and question.strip():
        sections = [section_choices[s] for s in chosen_sections] or None

        try:
            engine = load_engine()
        except PipelineNotBuilt as exc:
            # A skipped build step should read as setup instructions, not as a
            # crash dump in the middle of the page.
            st.error("The index has not been built yet.")
            st.code(str(exc), language="text")
            st.stop()

        spinner = "Searching..." if retrieval_only else "Searching the filings..."
        with st.spinner(spinner):
            try:
                if retrieval_only:
                    result = engine.retrieve(question, tickers or None, sections, top_k)
                else:
                    result = engine.query(question, tickers or None, sections, top_k)
            except Exception as exc:
                st.error(f"{type(exc).__name__}: {exc}")
                st.stop()

        companies = [paths.company_name(t) for t in result["tickers"]]
        st.caption("Searched " + (", ".join(companies) if companies
                                  else "all five companies")
                   + " · " + (", ".join(paths.section_label(x)
                                        for x in result["sections"])
                              or "Risk Factors and MD&A"))

        if "answer" in result:
            st.subheader("Answer")
            st.markdown(result["answer"])
            st.caption("Each [Source N] points to the matching passage below.")

        st.subheader("Sources")
        for i, src in enumerate(result["sources"], 1):
            year = paths.filing_year(src["filing_id"])
            label = (f"{i}. {paths.company_name(src['ticker'])}"
                     f"{f' · {year} 10-K' if year else ''}"
                     f" · {paths.section_label(src['section'], short=True)}")
            with st.expander(label):
                st.write(src["text"])
                st.caption(f"SEC accession {src['filing_id']}")

        render_analysis(question, result)

# --------------------------------------------------------------- How it works

with how_tab:
    st.subheader("Architecture")
    # st.markdown sanitises <svg> away, so embed the diagram in an iframe.
    st.iframe(ARCHITECTURE_SVG, height=250)

    st.markdown("""
The pipeline splits into two halves. The **top row runs once** to build the
index. The **bottom row runs on every question**.
""")

    st.divider()
    st.subheader("Building the index")

    st.markdown("""
**1 · Ingest** — `ingest_sec_data.py` pulls the last three 10-K filings for each
of the five companies from SEC EDGAR: 15 filings in total.

**2 · Parse and chunk** — `parse_documents.py` strips the HTML, then locates two
sections that carry most of the analytical content:

- **Item 1A — Risk Factors**: what management says could go wrong
- **Item 7 — MD&A**: management's own narrative on the numbers

Finding those boundaries is the hardest part of the whole pipeline. A phrase
like *"Item 1A. Risk Factors"* appears many times in a 10-K — once in the table
of contents and repeatedly as cross-references such as *"see Item 1A. Risk
Factors of this Form 10-K"*. Matching the wrong one silently files MD&A text
under risk factors. Each candidate heading is therefore rejected if it is
followed by a page number and the next item (a contents entry) or by punctuation
or a phrase like *"of this Form 10-K"* (a cross-reference). Headings are also
matched letter-by-letter with optional gaps, because HTML styling splits words
apart — Microsoft's filings render the heading as `ITEM 1A. RIS K FACTORS`.

The extracted text is cut into **150-word passages with 50 words of overlap**, so
a sentence spanning a boundary still lands intact in one of them.

**3 · Embed and store** — `build_vector_db.py` turns every passage into a vector
with `bge-small-en-v1.5`, which runs locally at no API cost, and persists them in
ChromaDB using cosine similarity.
""")

    st.divider()
    st.subheader("Answering a question")

    st.markdown("""
**1 · Scope to a company.** If the question names one — *Apple*, *Azure*, *AWS*,
*CUDA* — retrieval is restricted to that company's filings. Without this a
question about one company readily surfaces another's text, because risk
language is near-identical across big tech. If the filter matches nothing, the
search falls back to the whole corpus rather than answering with no context.

**2 · Search twice, two different ways.** The two retrievers fail in opposite
directions, so both run independently over the top 20 candidates each:

- **Dense** (ChromaDB) matches on *meaning*. It finds a passage about *"supply
  chain disruption"* for a question about *"manufacturing problems"*, but can
  miss an exact term it has no strong embedding for.
- **Sparse** (BM25) matches on *words*. It nails specific identifiers — an export
  control classification like `3A090.a`, a product name — where embeddings blur,
  but misses paraphrases entirely.

**3 · Fuse the two rankings (RRF).** Reciprocal Rank Fusion scores each passage
as the sum of `1 / (60 + rank)` across the lists it appears in. A passage both
retrievers rank highly beats one that only a single retriever liked — agreement
is the signal. It uses ranks rather than scores, which is what makes it safe to
combine a cosine similarity and a BM25 score that are not on a comparable scale.

**4 · Re-rank with a cross-encoder.** Fusion is cheap and approximate. The
cross-encoder reads the question and each passage *together* rather than
comparing two independent vectors, which is far more accurate but too slow to run
over 2,700 passages. Running it only on the fused shortlist gets the accuracy at
a workable cost. The top 5 survive.

**5 · Generate a cited answer.** Those 5 passages, each tagged with its ticker,
filing ID and section, become the context. Gemini is instructed to use only that
context and to cite it as `[Source N]`, so every claim is traceable to a filing.
""")

    st.divider()
    st.subheader("Why answers stay checkable")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
**Grounding**
- The model sees only the retrieved passages
- It is told never to invent figures or dates
- It is told to say so when context is insufficient
- Every passage is shown alongside the answer
""")
    with col_b:
        st.markdown("""
**Verification**
- `evaluate_retrieval.py` scores retrieval with no LLM, so it costs no quota
- Hit Rate, MRR, Precision@k, and company-detection accuracy
- `evaluate.py` adds RAGAS answer-quality metrics when quota allows
- 82 unit tests cover the boundary detection, fusion and filter logic
""")

    st.info("**Retrieval only** in the sidebar skips step 5 entirely. The Gemini "
            "free tier allows 20 generations per day, so this is the way to "
            "explore and tune retrieval without spending any of them.")

# -------------------------------------------------------------------- Corpus

with corpus_tab:
    rows, sections = corpus_stats()

    if not rows:
        st.warning("No parsed filings found. Run the ingest, parse and build "
                   "steps first.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Companies", len({r["Company"] for r in rows}))
        c2.metric("Annual reports", len(rows))
        c3.metric("Risk Factors passages", f"{sections['Item_1A_Risk_Factors']:,}")
        c4.metric("MD&A passages", f"{sections['Item_7_MDA']:,}")

        st.caption("The three most recent annual reports for each company. "
                   "Every report should contribute passages to both sections — "
                   "a zero would mean the text could not be located in it.")
        st.dataframe(sorted(rows, key=lambda r: (r["Company"], r["Filed"])),
                     width="stretch", hide_index=True)
