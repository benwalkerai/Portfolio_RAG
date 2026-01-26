# Portfolio Project: Autonomous RAG Agent

## Project Overview
This project demonstrates a clean, shippable Retrieval-Augmented Generation (RAG) pipeline. It is built using FastAPI for the API layer and LangChain for orchestration. The goal is to provide accurate answers based on a specific document corpus with full citations.

## Technical Stack
* **Orchestration**: LangChain
* **Vector Store**: Qdrant
* **API**: FastAPI
* **Embeddings**: OpenAI-compatible models

## Key Features
1. **Multi-format Ingestion**: Supports TXT, MD, and PDF files.
2. **Deterministic Citations**: Returns the source, document ID, and chunk index for every answer.
3. **Evaluation Loop**: Includes a script to compute Recall@k and measure retrieval quality.

## Deployment
The application is containerized using Docker, specifically running a Qdrant instance for vector storage. The ingestion process can be triggered via a CLI command to update the index without rebuilding the entire stack.