import os
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

def get_vector_store():
    """
    Returns a configured QdrantVectorStore instance using settings from environment variables.
    """
    qdrant_url = os.getenv("QDRANT_URL")
    collection_name = os.getenv("QDRANT_COLLECTION_NAME")
    embedding_model = os.getenv("EMBEDDING_MODEL_NAME")
    base_url = os.getenv("OPENAI_BASE_URL")

    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        openai_api_base=base_url,
        api_key=os.getenv("OPENAI_API_KEY"),
        check_embedding_ctx_length=False
    )

    client = QdrantClient(url=qdrant_url)

    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings
    )
