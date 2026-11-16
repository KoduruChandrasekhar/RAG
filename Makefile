# FilingScope task runner.
#   make build    - run the whole pipeline from scratch
#   make eval     - retrieval metrics (no API quota)
#   make ui       - launch the Streamlit app
# On Windows without make, run the python commands directly; see the README.

PY ?= python

.PHONY: help install ingest parse index build ask ui test test-fast eval ablate labels eval-ragas clean-index

help:
	@echo "install     install pinned dependencies"
	@echo "ingest      download 10-K filings from SEC EDGAR"
	@echo "parse       extract Item 1A / Item 7 and chunk them"
	@echo "index       embed the chunks into ChromaDB"
	@echo "build       ingest + parse + index"
	@echo "ask Q=...   ask one question"
	@echo "ui          launch the Streamlit app"
	@echo "test        full test suite"
	@echo "test-fast   parsing tests only (~1s, skips the torch import)"
	@echo "eval        retrieval metrics, no LLM calls, no API quota"
	@echo "ablate      measure what each retrieval stage contributes"
	@echo "labels      build the passage-labelling tool (enables nDCG)"
	@echo "eval-ragas  RAGAS answer metrics (needs Gemini quota)"
	@echo "clean-index drop parsed data and embeddings"

install:
	$(PY) -m pip install -r requirements.txt

ingest:
	$(PY) src/ingest_sec_data.py

parse:
	$(PY) src/parse_documents.py

index:
	$(PY) src/build_vector_db.py

build: ingest parse index

# usage: make ask Q="What risks did Apple flag about China?"
ask:
	$(PY) src/query_engine.py $(Q)

ui:
	$(PY) -m streamlit run src/app.py

test:
	$(PY) -m pytest tests/ -q

test-fast:
	$(PY) -m pytest tests/test_parse_documents.py -q

eval:
	$(PY) src/evaluate_retrieval.py

ablate:
	$(PY) src/ablation.py

labels:
	$(PY) src/build_label_pool.py

# Keep this small: the Gemini free tier allows 20 requests per day.
eval-ragas:
	$(PY) src/evaluate.py --limit 2

clean-index:
	rm -rf data/parsed data/embeddings
