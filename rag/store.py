# Import modules
import os
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_vector_store():
    """
    Returns the Qdrant Vector store instance.
    Used by both the Streamlist App and the Eval pipeline.
    """

    qdrant_url = os.getenv("QDRANT_URL")
    collection_name = os.getenv("QDRANT_COLLECTION_NAME")
    embedding_model = os.getenv("EMBEDDING_MODEL_NAME")
    base_url = os.getenv("OPENAI_BASE_URL")

    # Init Embeddings
    embeddings = OpenAIEmbeddings(
        model = embedding_model,
        openai_api_base=base_url,
        api_key=os.getenv("OPENAI_API_KEY"),
        check_embedding_ctx_length=False
    )

    # Init Client
    client = QdrantClient(url=qdrant_url)

    # Return Store
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings
    )

