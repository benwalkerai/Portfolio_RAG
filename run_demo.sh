#!/bin/bash
echo "[0/3] Note: Using in-memory Qdrant (no Docker required)"

echo "[1/3] Ingesting data..."
uv run python -m rag.ingest --path data/ --recreate true

echo "[2/3] Running Evals..."
uv run python evals/run_eval.py

echo "[3/3] Launching Chat App..."
uv run streamlit run app.py