#!/bin/bash
docker compose up -d
uv run python rag/ingest.pyt --path data/ --recreate true
uv run python evals/run_eval.py
uv run streamlit run app.py