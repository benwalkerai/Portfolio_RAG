# Import modules
import pytest
import os
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
TEST_COLLECTION = "test_portfolio_retrieval"
QDRANT_URL = os.getenv("QDRANT_URL")

@pytest.fixture
def vector_store():
    """Setup: Create a clean vector store for test."""
    client = QdrantClient(url=QDRANT_URL)

    if client.collection_exists(TEST_COLLECTION):
        client.delete_collection(TEST_COLLECTION)
    
    client.create_collection(
        collection_name=TEST_COLLECTION,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )

    embeddings = OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL_NAME"),
        openai_api_base=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        check_embedding_ctx_length=False
    )

    store = QdrantVectorStore(
        client=client,
        collection_name=TEST_COLLECTION,
        embedding=embeddings
    )

    yield store

    # Cleanup after test
    client.delete_collection(TEST_COLLECTION)

def test_retrieval_golden_path(vector_store):
    """
    The Golden Path:
    1. Ingest a known 'needle' fact.
    2. Query for it
    3. Assert it is retrieved
    """

    # Ingest
    needle_text = "The project code name is Project Blueberry."
    meta = {"source": "secret_memo.txt", "chunk_index": 0}

    vector_store.add_texts(
        texts=[needle_text],
        metadatas=[meta]
    )

    # Retrieve
    retriever = vector_store.as_retriever(search_kqargs={"k": 1})
    results = retriever.invoke("What is the code name?")

    # Assert
    assert len(results) > 0, "Retriever returned no results"
    top_result = results[0]

    print(f"\nRetrieved: {top_result.page_content}")

    assert "Blueberry" in top_result.page_content
    assert top_result.metadata["source"] == "secret_memo.txt"


def test_sanity():
    assert 1+1 == 2