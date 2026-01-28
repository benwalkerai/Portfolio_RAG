import pytest
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

load_dotenv()

TEST_COLLECTION = "test_portfolio_retrieval"
QDRANT_URL = os.getenv("QDRANT_URL")

@pytest.fixture
def vector_store():
    """Setup: Create a clean vector store for testing."""
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

    client.delete_collection(TEST_COLLECTION)

def test_retrieval_golden_path(vector_store):
    """
    Verifies that a specific 'needle' fact can be retrieved by an exact query.
    """
    needle_text = "The project code name is Project Blueberry."
    meta = {"source": "secret_memo.txt", "chunk_index": 0}

    vector_store.add_texts(
        texts=[needle_text],
        metadatas=[meta]
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 1})
    results = retriever.invoke("What is the code name?")

    assert len(results) > 0, "Retriever returned no results"
    top_result = results[0]

    assert "Blueberry" in top_result.page_content
    assert top_result.metadata["source"] == "secret_memo.txt"
