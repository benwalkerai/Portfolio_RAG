import os
import uuid
import argparse
from typing import List
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

load_dotenv()

def run_ingestion(path: str, recreate: bool):
    loaders = {
        ".txt": TextLoader,
        ".md": TextLoader,
        ".pdf": PyPDFLoader,
    }

    docs = []
    for ext, loader_cls in loaders.items():
        loader = DirectoryLoader(path, glob=f"**/*{ext}", loader_cls=loader_cls)
        docs.extend(loader.load())
    
    if not docs:
        print(f"No documents found in {path}")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(docs)

    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        # Deterministic ID based on source content
        chunk.metadata["doc_id"] = str(uuid.uuid5(uuid.NAMESPACE_DNS, source))
        chunk.metadata["chunk_index"] = i

    model_name = os.getenv("EMBEDDING_MODEL_NAME", "nomic-embed-text")
    base_url = os.getenv("OPENAI_BASE_URL")

    embeddings = OpenAIEmbeddings(
        model=model_name,
        openai_api_base=base_url,
        api_key=os.getenv("OPENAI_API_KEY"),
        check_embedding_ctx_length=False
    )

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection_name = os.getenv("QDRANT_COLLECTION_NAME", "portfolio_docs")

    client = QdrantClient(url=qdrant_url)

    if recreate:
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE) # Assuming 768-dimensional embeddings, adjust if using a different model
        )
    
    QdrantVectorStore.from_documents(
        chunks,
        embeddings,
        url=qdrant_url,
        collection_name=collection_name,
    )
    print(f"Successfully indexed {len(chunks)} chunks from {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="data/")
    parser.add_argument("--recreate", type=str, default="false")
    args = parser.parse_args()

    run_ingestion(args.path, args.recreate.lower() == "true")
