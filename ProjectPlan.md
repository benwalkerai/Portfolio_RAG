# Project Plan - Portfolio RAG (Simple, Solid, Shippable)

## Goal
Ship a working RAG chat application that demonstrates a clean LangChain-based RAG pipeline (ingest -> index -> retrieve -> generate) with citations and a lightweight eval loop.

---

## Tech Stack (for v1)
- API: FastAPI
- Orchestration: LangChain
- Vector Store: Qdrant + 'langchain_qdrant'
- Embeddings + LLM: OpenAI-compativle providers (config-driven)
- Optional eval/observability: LangSmith

---

## Architecture'
### Core Pipeline (LangChain)
1. Ingest:
    - Load docs using 'langchain_community.document_loaders' (TXT, MD & PDF)
2. Split:
    - Chunk with 'RecursiveCharacterTextSplitter' (store chunk metadata)
3. Index:
    - Create/update Qdrant collect using 'QdrantVectorStore.from_documents(...)'.
4. Retrieve:
    - Use 'vector_store.as_retriever(...)' and start with vanilla similarity;
5. Generate:
    - Build an LCEL chain: (question -> retriever -> prompt -> LLM -> parse output) and return citations.

### Endpoints
- 'POST /ingest' (or CLI): loads 'data/' and indexes.
- 'POST /chat': runs the chain and returns '{answer, citations}'.

---

## Milestones
## M0 - Minimal runnable skeleton
**Outcome:** Runs locally in minutes.

Tasks
- Create slim layout:
    - 'api/' (FastAPI)
    - 'rag/' (langchain wiring)
    - 'tests/'
    - 'data/'
    - 'docker/' (qdrant only)
- Keep ONE 'docker-compose.yml
- Add '.env.example' with minimal config:
    - LLM/embedding provider keys
    - Qdrant URL + collection name

Deliverables
- README quickstart:
    - 'docker compose up -d'
    - 'uv run python -m ...'
    - curl example for '/chat'

---

## M1 - Ingest + index with LangChain
**Outcome:** Documents load, chunk and index into Qdrant.

Tasks
- Implement 'rag/ingest.py':
    - Load from 'data/' (TXT, MD, PDF)
    - Add metadata fields: 'source', 'doc_id', 'chunk_index'
- Implement 'rag/split.py':
    - 'RecursiveCharacterTextSplitter' with config constants
- Implement 'rag/index.py':
    - 'QdrantVectorStore.from_documents(docs, embedding=..., collection_name=...)' [LangChain integration]

Deliverables
- Command: 'python -m rag.ingest --path data/ --recreate false'

---

## M2 - Retrieval + citations
**Outcome:** Retrieval works and citations are visible and consistent.

Tasks
- Implement 'rag/retrieve.py':
    - 'retriever = vector_store.as_retriever(search_kwargs={"k": K}); [Qdrant retriever]
- Implement citation formatting:
    - Return citations as structured objects:
        - '{source, doc_id, chunk_index, snippet}'
- Add a deterministic fixture doc and a query that should retrieve a specific chunk.

Deliverables
- 'tests/test_retrieve.py': sanity check top-k contains expected 'doc_id'/keyword.
- tests/test_citations.py': ensures citations include required fields.

---

## M3 - Chat chain (LCEL) + API
**Outcome:** '/chat' runs end-to-end and returns answer + citations.

Tasks
- Implement 'rag/chain.py' using LCEL:
    - input question
    - call retriever
    - compose prompt with retrieved chunkn texts
    - call LLM
    - output parser returns '{answer, citations}'

Deliverables
- 'POST /chat' example in README

---

## M4 - Evaluation Loop
**Outcome:** You can measure retrieval quality and show iteration.

Tasks
- Add 'evals/' dataset with 10-30 Q/A items:
    - minimal labels: expected 'source' or expected keyword in retrieved context
- Add eval script:
    - computes Recall@k (and optional MRR)
- Document baseline result in README.

Deliverables
- 'evals/run.py' + 'evals/dataset.json'

---

## M5 - One measured improvement
**Outcome:** Demonstrate engineering judgement: change _> metric improvement.

Pick ONE:
- Switch retriever search mode to MMR (often improves diversity). [LangChain Qdrant docs show MMR usage] 
- Improve chunk size/overlap.
- Add metadata filtering (e.g., restrict to doc_id if user selects a document).

Deliverables
- Before/after metrics table in README.

---

## Risks and mitigations
- Scope creep: any feature that requires new services or auth goes to v2.
- Flaky generation: evaluate retrieval separately; don’t overfit to “LLM answer grading” in v1.
- Debug drag: keep a “golden path” script: ingest sample docs → run 3 queries → print citations.

---

## v2 Ideas (optional)
- Hybrid retrieval (dense + sparse) once you can prove improvement.
- LangSmith traces + dataset-driven evals.
- Streaming responses and basic conversation memory (only after single-turn is rock solid).