import os
from pydantic import SecretStr
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

def get_vector_store():
    """
    Returns a configured QdrantVectorStore instance using settings from environment variables.
    Supports both remote Qdrant (Docker) and local file-based storage modes.
    """
    collection_name = os.getenv("QDRANT_COLLECTION_NAME", "portfolio_docs")
    embedding_model = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")
    use_local_storage = os.getenv("QDRANT_IN_MEMORY", "false").lower() == "true"
    
    # Get API key (required)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    # Workaround for OpenAI Windows platform detection issue
    import platform as platform_module
    original_system = platform_module.system
    try:
        # Wrap platform.system() to avoid WMI errors on Windows
        platform_module.system = lambda: "Windows"
        embeddings = OpenAIEmbeddings(
            model=embedding_model,
            api_key=SecretStr(api_key),
            check_embedding_ctx_length=False
        )
    finally:
        platform_module.system = original_system

    if use_local_storage:
        storage_path = "./qdrant_storage"
        os.makedirs(storage_path, exist_ok=True)
        client = QdrantClient(path=storage_path)
    else:
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        client = QdrantClient(url=qdrant_url)

    try:
        return QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings
        )
    except Exception as e:
        # If validation fails (e.g., collection doesn't exist), create it
        if "not found" in str(e).lower() or "does not exist" in str(e).lower():
            from qdrant_client.http.models import Distance, VectorParams
            # Create collection with reasonable default dimension (1536 for text-embedding-3-small)
            if not client.collection_exists(collection_name):
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
                )
            return QdrantVectorStore(
                client=client,
                collection_name=collection_name,
                embedding=embeddings
            )
        raise
