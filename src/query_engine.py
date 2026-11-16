import os
import re
import sys
import json
import time
import argparse

import httpx
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from dotenv import load_dotenv

import paths

load_dotenv()

# httpx backs the google-genai client. A dropped connection or read timeout is
# transient and must be retried, not allowed to kill a long evaluation run.
TRANSIENT_NETWORK_ERRORS = (httpx.TransportError,)

# Word-overlap above this counts as the same passage restated. Calibrated
# against the corpus: real near-duplicates score 0.75+, unrelated pairs ~0.18.
DUPLICATE_THRESHOLD = 0.70


class PipelineNotBuilt(RuntimeError):
    """The index or parsed data is missing, so a build step has not been run."""


class FilingScopeEngine:
    """
    Stage 4 Engine: Hybrid Retrieval (Dense ChromaDB + BM25 Sparse via RRF)
    + Cross-Encoder Re-Ranker + Gemini generation.
    """

    @staticmethod
    def _check_index_exists():
        # Without this the failure surfaces as an opaque ChromaDB or listdir
        # error that gives no hint which build step was skipped.
        if not os.path.isdir(paths.PARSED_DIR) or not [
                f for f in os.listdir(paths.PARSED_DIR) if f.endswith(".json")]:
            raise PipelineNotBuilt(
                f"No parsed filings in {paths.PARSED_DIR}.\n"
                f"Run:  python src/ingest_sec_data.py\n"
                f"      python src/parse_documents.py\n"
                f"      python src/build_vector_db.py"
            )
        if not os.path.isdir(paths.EMBEDDINGS_DIR):
            raise PipelineNotBuilt(
                f"No vector store in {paths.EMBEDDINGS_DIR}.\n"
                f"Run:  python src/build_vector_db.py"
            )

    def __init__(self):
        self._check_index_exists()
        print("Loading FilingScope engine...")

        # 1. ChromaDB (Dense Semantic Search)
        self.client = chromadb.PersistentClient(path=paths.EMBEDDINGS_DIR)
        self.ef = SentenceTransformerEmbeddingFunction(
            model_name=paths.EMBEDDING_MODEL
        )
        try:
            self.collection = self.client.get_collection(
                name=paths.COLLECTION_NAME,
                embedding_function=self.ef
            )
        except Exception as exc:
            raise PipelineNotBuilt(
                f"No '{paths.COLLECTION_NAME}' collection in "
                f"{paths.EMBEDDINGS_DIR}.\n"
                f"Run:  python src/build_vector_db.py\n"
                f"(underlying error: {type(exc).__name__}: {exc})"
            ) from exc

        # 2. Cross-Encoder Re-Ranker
        print("Loading re-ranker model...")
        self.reranker = CrossEncoder(paths.RERANKER_MODEL)

        # 3. BM25 (Sparse Keyword Search)
        print("Building BM25 index...")
        self.all_chunks, self.bm25 = self._build_bm25_index()

        # 4. Gemini Client. Optional: without a key the engine still retrieves,
        # it just cannot write an answer. A public demo may well run this way
        # rather than expose someone's quota.
        api_key = os.getenv("GEMINI_API_KEY")
        self.gemini = genai.Client(api_key=api_key) if api_key else None
        if not self.gemini:
            print("  No GEMINI_API_KEY set -- retrieval only, no answer generation.")

        print(f"Engine ready. Chunks in database: {self.collection.count()}")

    @property
    def can_generate(self):
        return self.gemini is not None

    def _build_bm25_index(self):
        parsed_dir = paths.PARSED_DIR
        all_chunks = []

        for filename in sorted(os.listdir(parsed_dir)):
            if not filename.endswith(".json"):
                continue
            filepath = os.path.join(parsed_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            all_chunks.extend(data["chunks"])

        tokenized = [chunk["text"].lower().split() for chunk in all_chunks]
        bm25 = BM25Okapi(tokenized)

        print(f"  BM25 index built: {len(all_chunks)} chunks indexed")
        return all_chunks, bm25

    def _detect_tickers(self, question):
        # A question that names a company should not be answered from another
        # company's filing, so scope retrieval when we can tell who is meant.
        #
        # Matching must respect word boundaries. A plain substring test reads
        # "aws" inside "l-aws" and "mac" inside "mac-hine", so "export control
        # laws" scoped the whole search to Amazon and "machine learning" pulled
        # in Apple.
        asked = question.lower()
        found = [t for t, names in paths.TICKER_ALIASES.items()
                 if any(re.search(rf"\b{re.escape(name)}\b", asked)
                        for name in names)]
        return found

    @staticmethod
    def _build_where(tickers=None, sections=None):
        # Chroma wants a bare equality for one value, $in for several, and the
        # clauses wrapped in $and when filtering on more than one field.
        clauses = []
        for field, values in (("ticker", tickers), ("section", sections)):
            if not values:
                continue
            if len(values) == 1:
                clauses.append({field: values[0]})
            else:
                clauses.append({field: {"$in": list(values)}})

        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def _dense_retrieve(self, question, n=20, tickers=None, sections=None):
        results = self.collection.query(
            query_texts=[question],
            n_results=n,
            where=self._build_where(tickers, sections)
        )

        chunks = []
        for i, doc in enumerate(results["documents"][0]):
            chunks.append({
                "text": doc,
                "metadata": results["metadatas"][0][i],
                "id": results["ids"][0][i],
                "dense_rank": i + 1
            })
        return chunks

    def _sparse_retrieve(self, question, n=20, tickers=None, sections=None):
        tokenized_query = question.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        candidates = range(len(scores))
        if tickers:
            allowed = set(tickers)
            candidates = [i for i in candidates
                          if self.all_chunks[i]["ticker"] in allowed]
        if sections:
            allowed = set(sections)
            candidates = [i for i in candidates
                          if self.all_chunks[i]["section"] in allowed]

        top_indices = sorted(
            candidates,
            key=lambda i: scores[i],
            reverse=True
        )[:n]

        chunks = []
        for rank, idx in enumerate(top_indices):
            chunk = self.all_chunks[idx]
            chunks.append({
                "text": chunk["text"],
                "metadata": {
                    "ticker": chunk["ticker"],
                    "filing_id": chunk["filing_id"],
                    "section": chunk["section"]
                },
                "id": chunk["chunk_id"],
                "sparse_rank": rank + 1
            })
        return chunks

    @staticmethod
    def _fuse(dense_chunks, sparse_chunks):
        # Reciprocal Rank Fusion. Scoring by rank rather than by raw score is
        # what makes it safe to merge a cosine similarity with a BM25 score,
        # which are not on comparable scales.
        k = 60
        rrf_scores = {}
        chunk_map = {}

        for chunk in dense_chunks:
            key = chunk["id"]
            rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (k + chunk["dense_rank"])
            chunk_map[key] = chunk

        for chunk in sparse_chunks:
            key = chunk["id"]
            rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (k + chunk["sparse_rank"])
            if key not in chunk_map:
                chunk_map[key] = chunk

        sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        return [chunk_map[key] for key in sorted_keys[:40]]

    def _hybrid_retrieve(self, question, n_each=20, tickers=None, sections=None):
        dense_chunks = self._dense_retrieve(
            question, n=n_each, tickers=tickers, sections=sections)
        sparse_chunks = self._sparse_retrieve(
            question, n=n_each, tickers=tickers, sections=sections)
        return self._fuse(dense_chunks, sparse_chunks)

    def _rerank(self, question, chunks, top_k=5):
        if not chunks:
            return []
        pairs = [(question, chunk["text"]) for chunk in chunks]
        scores = self.reranker.predict(pairs)

        ranked = sorted(
            zip(scores, chunks),
            key=lambda x: x[0],
            reverse=True
        )
        return [chunk for score, chunk in ranked[:top_k]]

    @staticmethod
    def _overlap(text_a, text_b):
        # Jaccard over word sets. Measured against the corpus, genuine
        # near-duplicates score >= 0.75 while unrelated passages sit near 0.18,
        # so DUPLICATE_THRESHOLD separates them cleanly.
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        if not words_a or not words_b:
            return 0.0
        return len(words_a & words_b) / len(words_a | words_b)

    @classmethod
    def _diversify(cls, ranked_chunks, top_k, threshold=DUPLICATE_THRESHOLD):
        """Drop passages that repeat one already selected.

        Filings restate the same boilerplate year after year, so without this
        the top results fill up with the same paragraph from three different
        annual reports -- crowding out other evidence and wasting the context
        the model gets to read.
        """
        kept, skipped = [], []
        for chunk in ranked_chunks:
            duplicate = any(cls._overlap(chunk["text"], k["text"]) >= threshold
                            for k in kept)
            (skipped if duplicate else kept).append(chunk)
            if len(kept) == top_k:
                break

        # Better to return a near-duplicate than fewer passages than asked for.
        if len(kept) < top_k:
            kept.extend(skipped[:top_k - len(kept)])
        return kept[:top_k]

    @classmethod
    def _balance_by_company(cls, ranked_chunks, top_k, tickers,
                            threshold=DUPLICATE_THRESHOLD):
        """Give every company in scope a share of the slots.

        For a comparative question ("how do X and Y each describe Z"), the
        re-ranker often judges one company's passages more on-topic and fills
        every slot with them -- leaving nothing to compare against. Reserving a
        quota per company keeps both sides of the comparison present.
        """
        if len(tickers) < 2:
            return cls._diversify(ranked_chunks, top_k, threshold)

        quota = max(1, top_k // len(tickers))
        kept = []

        def is_duplicate(chunk):
            return any(cls._overlap(chunk["text"], k["text"]) >= threshold
                       for k in kept)

        # Fill each company's quota first, best-ranked passage first.
        for ticker in tickers:
            taken = 0
            for chunk in ranked_chunks:
                if taken >= quota or len(kept) >= top_k:
                    break
                if chunk["metadata"]["ticker"] != ticker or is_duplicate(chunk):
                    continue
                kept.append(chunk)
                taken += 1

        # Any slots left go to the best remaining passages, whoever they are.
        chosen = {id(c) for c in kept}
        for chunk in ranked_chunks:
            if len(kept) >= top_k:
                break
            if id(chunk) in chosen or is_duplicate(chunk):
                continue
            kept.append(chunk)
            chosen.add(id(chunk))

        # Present them in the re-ranker's order, not grouped by company.
        order = {id(c): i for i, c in enumerate(ranked_chunks)}
        return sorted(kept[:top_k], key=lambda c: order[id(c)])

    def _generate_answer(self, question, chunks):
        context_str = ""
        for idx, chunk in enumerate(chunks):
            meta = chunk["metadata"]
            context_str += (
                f"\n--- Source [{idx+1}]: "
                f"Ticker: {meta['ticker']} | "
                f"Filing: {meta['filing_id']} | "
                f"Section: {meta['section']} ---\n"
            )
            context_str += chunk["text"] + "\n"

        prompt = f"""You are an expert financial analyst.
Answer the question using ONLY the context provided from SEC 10-K filings.
Cite sources using [Source X] format where appropriate.
Never invent numbers, dates, or facts not present in the context.
If context is insufficient, say so clearly.

Context:
{context_str}

Question: {question}

Answer:"""

        config = types.GenerateContentConfig(
            temperature=0.1,
            # gemini-3.x flash is a thinking model: reasoning tokens are
            # billed against max_output_tokens. At 500 the reasoning pass
            # consumed ~478 of them and answers were cut off mid-sentence
            # (finish_reason=MAX_TOKENS), so leave room for both.
            max_output_tokens=2048
        )

        if not self.can_generate:
            return ("No GEMINI_API_KEY is configured, so no answer was written. "
                    "The passages below are the ones that matched your question.")

        response = self._generate_with_retry(prompt, config)

        # .text is None when a response is empty or blocked; returning None
        # breaks every caller downstream, so say so instead.
        if response is None:
            return "Generation failed: the model was unavailable after retries."
        if not response.text:
            return "Generation returned no text (the response was empty or blocked)."

        return response.text

    def _generate_with_retry(self, prompt, config, attempts=5):
        # The free tier returns 503 UNAVAILABLE on load spikes and 429
        # RESOURCE_EXHAUSTED on rate limits, and the connection itself drops
        # often enough to matter. All are transient, and without this an
        # evaluation run dies partway through.
        delay = 5
        for attempt in range(1, attempts + 1):
            try:
                return self.gemini.models.generate_content(
                    model=paths.GEMINI_MODEL,
                    contents=prompt,
                    config=config
                )
            except TRANSIENT_NETWORK_ERRORS as exc:
                if attempt == attempts:
                    print(f"  Network unreachable after {attempts} attempts: "
                          f"{type(exc).__name__}")
                    return None
                print(f"  Network error ({type(exc).__name__}); "
                      f"retry {attempt}/{attempts - 1} in {delay}s")
            except genai_errors.APIError as exc:
                retryable = getattr(exc, "code", None) in (429, 500, 503)
                if not retryable or attempt == attempts:
                    if not retryable:
                        raise
                    print(f"  Gemini unavailable after {attempts} attempts: {exc.code}")
                    return None
                print(f"  Gemini {exc.code}; retry {attempt}/{attempts - 1} in {delay}s")
                time.sleep(delay)
                delay = min(delay * 2, 60)
        return None

    def _gather(self, question, tickers, sections, n_each):
        """Collect fused candidates, searching per company when several are in
        scope.

        A pooled search across two companies is won by whichever one the
        question suits better: asking how Microsoft and NVIDIA each describe
        export controls returns 31 NVIDIA passages to 5 Microsoft ones, and
        re-ranking then strips Microsoft out altogether. Searching once per
        company gives each its own shortlist to compete from.
        """
        if tickers and len(tickers) > 1:
            candidates, dense_n, sparse_n = [], 0, 0
            for ticker in tickers:
                dense = self._dense_retrieve(question, n=n_each,
                                             tickers=[ticker], sections=sections)
                sparse = self._sparse_retrieve(question, n=n_each,
                                               tickers=[ticker], sections=sections)
                candidates.extend(self._fuse(dense, sparse))
                dense_n += len(dense)
                sparse_n += len(sparse)
            return candidates, dense_n, sparse_n

        dense = self._dense_retrieve(question, n=n_each, tickers=tickers,
                                     sections=sections)
        sparse = self._sparse_retrieve(question, n=n_each, tickers=tickers,
                                       sections=sections)
        return self._fuse(dense, sparse), len(dense), len(sparse)

    def retrieve(self, question, tickers=None, sections=None, top_k=5,
                 diversify=True):
        """Run retrieval only -- no LLM call, so this costs no API quota.

        tickers=None auto-detects the company named in the question; pass a
        list to force it, or [] to search the whole corpus.
        """
        detected = self._detect_tickers(question)
        if tickers is None:
            tickers = detected
            scope_source = "detected" if detected else "none"
        else:
            scope_source = "chosen" if tickers else "all"

        n_each = 20
        candidates, dense_n, sparse_n = self._gather(
            question, tickers, sections, n_each)

        widened = False
        if not candidates and (tickers or sections):
            # Nothing matched under the filter -- fall back to the full corpus
            # rather than answering with no context at all.
            candidates, dense_n, sparse_n = self._gather(
                question, [], None, n_each)
            widened = True

        # Re-rank a wider shortlist than needed, then pick a diverse top_k from
        # it -- there has to be slack for the duplicate filter to choose from.
        ranked = self._rerank(question, candidates, top_k=max(top_k * 4, top_k))
        if diversify:
            top_chunks = self._balance_by_company(ranked, top_k, tickers or [])
        else:
            top_chunks = ranked[:top_k]

        # How many passages the duplicate filter displaced from the plain
        # top_k. Chunks are dicts, so compare by id, not by value.
        kept_ids = {id(c) for c in top_chunks}
        displaced = sum(1 for c in ranked[:top_k] if id(c) not in kept_ids)

        return {
            "tickers": tickers,
            "sections": sections or [],
            # A record of what the pipeline did with this particular question,
            # so the answer can be explained without reading the code.
            "trace": {
                "detected_tickers": detected,
                "scope_source": scope_source,
                "dense_hits": dense_n,
                "sparse_hits": sparse_n,
                "fused_candidates": len(candidates),
                "reranked_shortlist": len(ranked),
                "duplicates_dropped": displaced,
                "kept": len(top_chunks),
                "widened_to_full_corpus": widened,
            },
            "sources": [
                {
                    # chunk_id is the stable key relevance labels are stored
                    # against, so it must survive out of retrieval.
                    "chunk_id": c.get("id"),
                    "ticker": c["metadata"]["ticker"],
                    "filing_id": c["metadata"]["filing_id"],
                    "section": c["metadata"]["section"],
                    "text": c["text"]
                }
                for c in top_chunks
            ]
        }

    def query(self, question, tickers=None, sections=None, top_k=5,
              diversify=True):
        """Retrieve, then generate a cited answer from what was retrieved."""
        result = self.retrieve(question, tickers, sections, top_k, diversify)
        chunks = [{"text": s["text"], "metadata": s} for s in result["sources"]]
        result["answer"] = self._generate_answer(question, chunks)
        return result


SECTION_CHOICES = {
    "risk": "Item_1A_Risk_Factors",
    "mda": "Item_7_MDA",
}


def format_result(result, show_text=False):
    lines = []
    scope = ", ".join(result["tickers"]) if result["tickers"] else "all companies"
    sections = ", ".join(result["sections"]) if result["sections"] else "all sections"
    lines.append(f"SCOPED TO: {scope}  |  {sections}")

    if "answer" in result:
        lines.append("")
        lines.append("=" * 60)
        lines.append("ANSWER")
        lines.append("=" * 60)
        lines.append(result["answer"])

    lines.append("")
    lines.append("=" * 60)
    lines.append("SOURCES")
    lines.append("=" * 60)
    for i, src in enumerate(result["sources"], 1):
        lines.append(f"  [{i}] {src['ticker']} | {src['filing_id']} | {src['section']}")
        if show_text:
            lines.append(f"      {src['text'][:300]}")
            lines.append("")
    return "\n".join(lines)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="query_engine",
        description="Ask a question about the indexed SEC 10-K filings.")
    parser.add_argument("question", nargs="*",
                        help="the question; omit it to enter interactive mode")
    parser.add_argument("-t", "--ticker", action="append", metavar="SYM",
                        choices=paths.TICKERS,
                        help="restrict to a ticker (repeatable). Default: "
                             "inferred from the question")
    parser.add_argument("-s", "--section", action="append",
                        choices=sorted(SECTION_CHOICES),
                        help="restrict to 'risk' (Item 1A) or 'mda' (Item 7)")
    parser.add_argument("-k", "--top-k", type=int, default=5, metavar="N",
                        help="how many chunks to keep after re-ranking (default 5)")
    parser.add_argument("-a", "--all-companies", action="store_true",
                        help="search every filing, skipping company detection")
    parser.add_argument("-r", "--retrieval-only", action="store_true",
                        help="show retrieved sources without calling the LLM "
                             "(uses no API quota)")
    parser.add_argument("-i", "--interactive", action="store_true",
                        help="keep asking questions without reloading the models")
    return parser


def run_once(engine, question, args):
    tickers = args.ticker
    if args.all_companies:
        tickers = []
    sections = [SECTION_CHOICES[s] for s in args.section] if args.section else None

    if args.retrieval_only:
        result = engine.retrieve(question, tickers, sections, args.top_k)
    else:
        result = engine.query(question, tickers, sections, args.top_k)

    print("\n" + "=" * 60)
    print(f"QUESTION: {question}")
    print(format_result(result, show_text=args.retrieval_only))


def interactive_loop(engine, args):
    print("\nInteractive mode. Blank line or Ctrl-C to quit.")
    if args.retrieval_only:
        print("Retrieval only -- no LLM calls, no API quota used.")
    while True:
        try:
            question = input("\nQuestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not question:
            return
        try:
            run_once(engine, question, args)
        except Exception as exc:
            # Keep the session alive; reloading the models takes ~20s.
            print(f"  Query failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    args = build_parser().parse_args()
    question = " ".join(args.question).strip()

    if not question and not args.interactive:
        # Nothing to answer, so make the session useful rather than erroring.
        args.interactive = True

    try:
        engine = FilingScopeEngine()
    except PipelineNotBuilt as exc:
        # A missing build step is a setup problem, not a crash: say which
        # command to run instead of printing a stack trace.
        print(f"\n{exc}\n")
        sys.exit(1)

    if question:
        run_once(engine, question, args)
    if args.interactive:
        interactive_loop(engine, args)
