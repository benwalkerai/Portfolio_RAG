# Import Modules
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(page_title="Portfolio RAG", layout="centered")
st.title("Ragbot")

# Cached Resources
@st.cache_resource
def get_vector_store():
    qdrant_url = os.getenv("QDRANT_URL")
    collection_name = os.getenv("QDRANT_COLLECTION_NAME")
    embedding_model = os.getenv("EMBEDDING_MODEL_NAME")
    base_url = os.getenv("OPENAI_BASE_URL")

    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        openai_api_base=base_url,
        check_embedding_ctx_length=False
    )

    client = QdrantClient(url=qdrant_url)

    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )

@st.cache_resource
def get_llm():
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL_NAME"),
        temperature=0.7,
        openai_api_base=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY", "dummy")
    )

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat History Display
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "citations" in message and message["citations"]:
            with st.expander("Sources"):
                for cite in message["citations"]:
                    st.markdown(f"**{cite['source']}**")
                    st.caption(cite['content'][:200] + "...")

# Chat Input & Processing
if prompt := st.chat_input("Ask a question about your docs..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
    
    # Retrieve
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(prompt)

    context_text = "\n\n".join([d.page_content for d in docs])
    system_prompt = f"""You are a helpful assistant. Answer the question based ONLY on the following context. If the answer is not in the context, say "I don't know."

    Context:
    {context_text}
    """

    # Generation
    llm = get_llm()
    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ])

    answer = response.content
    message_placeholder.markdown(answer)

    # Format Citations
    citations = []
    for doc in docs:
        citations.append({
            "source": os.path.basename(doc.metadata.get("source", "unknown")),
            "chunk_index": doc.metadata.get("chunk_index", "?"),
            "content": doc.page_content
        })


    with st.expander("Sources"):
        for cite in citations:
            st.markdown(f"**{cite['source']}** (Chunk {cite['chunk_index']})")
            st.caption(cite['content'][:200] + "...")

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "citations": citations
    })