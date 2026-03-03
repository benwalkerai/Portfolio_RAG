@echo off
echo [0/3] Note: Using local file-based Qdrant (no Docker required)

echo [1/3] Ingesting data...
uv run python -m rag.ingest --path data/ --recreate true
if errorlevel 1 (
    echo Error during ingestion. Continuing...
)
timeout /t 2 /nobreak

echo [2/3] Running Evals...
uv run python evals/run_eval.py
if errorlevel 1 (
    echo Error during evals. Continuing...
)
timeout /t 2 /nobreak

echo [3/3] Launching Chat App...
uv run streamlit run app.py