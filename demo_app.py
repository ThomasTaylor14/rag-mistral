"""
Streamlit app for the Financial Analyst RAG 
"""
import streamlit as st
from dotenv import load_dotenv
import os
from mistralai import Mistral
import numpy as np
import faiss
import time
import random
from typing import List

load_dotenv()

# Constants
INDEX_DIR = "faiss_index"
INDEX_FILE = f"{INDEX_DIR}/faiss_index.bin"
CHUNKS_FILE = f"{INDEX_DIR}/chunks.pkl"
ASSET_AVATAR = "assets/mistral-ai.svg"

st.set_page_config(page_title="Financial Analyst RAG", layout="centered")

st.sidebar.title("💬 Financial Analyst Chat")
st.sidebar.text("🔍 BNP Q2 2025 Earnings & Press Releases Assistant")

# Initialize Mistral client
MISTRAL_API_KEY = os.getenv("MISTRAL_API")

# Simple API key input if not found in environment
if not MISTRAL_API_KEY:
    with st.sidebar:
        st.warning("🔑 API Key Required")
        MISTRAL_API_KEY = st.text_input(
            "Enter your Mistral API Key:",
            type="password",
            help="Get your API key from https://console.mistral.ai/",
            key="api_key_input"
        )
        if not MISTRAL_API_KEY:
            st.info("Please enter your API key to continue")
            st.stop()

client = Mistral(api_key=MISTRAL_API_KEY)

# Simple API key verification
if not st.session_state.get("api_verified", False):
    try:
        # Test the API key by listing models (simpler and faster)
        client.models.list()
        st.session_state.api_verified = True
    except Exception as e:
        with st.sidebar:
            st.error(f"❌ Invalid API key: {str(e)}")
            if st.button("🔄 Try Again"):
                if "api_verified" in st.session_state:
                    del st.session_state["api_verified"]
                if "api_key_input" in st.session_state:
                    del st.session_state["api_key_input"]
                st.rerun()
        st.stop()

# Helper Functions
def get_embeddings_with_retry(
    inputs: List[str], max_retries: int = 5, batch_size: int = 10
):
    """
    Get embeddings with exponential backoff retry and batch processing
    """
    all_embeddings = []

    # Process in batches to reduce API calls
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i : i + batch_size]

        for attempt in range(max_retries):
            try:
                # Make API call for the batch
                embeddings_batch_response = client.embeddings.create(
                    model="mistral-embed", inputs=batch
                )

                # Extract embeddings from response
                batch_embeddings = [
                    item.embedding for item in embeddings_batch_response.data
                ]
                all_embeddings.extend(batch_embeddings)

                break

            except Exception as e:
                if "rate_limited" in str(e) or "429" in str(e):
                    if attempt < max_retries - 1:
                        # Exponential backoff with jitter
                        wait_time = (2**attempt) + random.uniform(0, 1)
                        time.sleep(wait_time)
                    else:
                        raise e
                else:
                    raise e

        # Small delay between batches to be respectful
        if i + batch_size < len(inputs):
            time.sleep(0.5)

    return all_embeddings

def format_chunk_with_source(chunk, chunk_num):
    """Format chunk with source information for better citations"""
    source = chunk.metadata.get("source", "Unknown Source")
    page = chunk.metadata.get("page", 0)
    display_page = page + 1 if isinstance(page, int) else page

    return f"""
--- CHUNK {chunk_num + 1} ---
Source: {source} (Page {display_page})
Content: {chunk.page_content}
---
"""

def create_source_summary(chunks):
    """Create a summary of sources used in the response"""
    sources = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "Unknown Source")
        page = chunk.metadata.get("page", 0)
        display_page = page + 1 if isinstance(page, int) else page

        if source not in sources:
            sources[source] = set()
        sources[source].add(display_page)

    summary = "**📚 SOURCES REFERENCED:**\n\n"
    for source, pages in sources.items():
        pages_str = ", ".join(
            map(str, sorted(pages) if all(isinstance(p, int) for p in pages) else pages)
        )
        summary += f"• **{source}**: Pages {pages_str}\n"

    return summary


def run_mistral(user_message, model="mistral-large-latest"):
    """Run Mistral AI chat completion"""
    messages = [{"role": "user", "content": user_message}]
    chat_response = client.chat.complete(model=model, messages=messages)
    return chat_response.choices[0].message.content


# Load pre-built FAISS index and data
@st.cache_data
def load_prebuilt_index():
    """Load pre-built FAISS index, embeddings, and chunks"""
    import pickle
    import os

    # Use constants for file paths
    index_file = INDEX_FILE
    chunks_file = CHUNKS_FILE

    # Check if required files exist
    if not all(os.path.exists(f) for f in [index_file, chunks_file]):
        return None, None, "files_not_found"

    try:
        # Load FAISS index (contains embeddings)
        similarity_index = faiss.read_index(index_file)

        # Load chunks with text content and metadata
        with open(chunks_file, "rb") as f:
            chunks = pickle.load(f)

        return similarity_index, chunks, "success"
    except Exception as e:
        return None, None, f"error: {str(e)}"


# Load the pre-built FAISS index and data (only on first run)
if 'index_loaded' not in st.session_state:
    with st.spinner("📁 Loading pre-built FAISS index..."):
        similarity_index, chunks, status = load_prebuilt_index()

        if status == "files_not_found":
            st.error(
                "❌ Pre-built index files not found! Please run the notebook first to create them."
            )
            st.info(f"Expected files: {INDEX_FILE}, {CHUNKS_FILE}")
            st.stop()
        elif status.startswith("error:"):
            st.error(f"❌ Error loading index: {status}")
            st.stop()
        else:
            st.toast(f"✅ Loaded pre-built index with {len(chunks)} chunks")
            
        # Store in session state for future use
        st.session_state.similarity_index = similarity_index
        st.session_state.chunks = chunks
        st.session_state.index_loaded = True
else:
    # Use data from session state (no loading UI)
    similarity_index = st.session_state.similarity_index
    chunks = st.session_state.chunks

# Update sidebar with document info and ready status
with st.sidebar:
    st.success("✅ Ready to chat!")
    
    with st.expander("📊 Document Info"):

        # Only show document info if index is loaded
        if st.session_state.get('index_loaded', False):
            # Document info from loaded chunks
            st.write(f"📄 **Total chunks:** {len(st.session_state.chunks)}")
            st.write("📚 **Sources:**")
            sources = [chunk.metadata.get("source", "Unknown") for chunk in st.session_state.chunks]
            for source in set(sources):
                count = sources.count(source)
                st.write(f"• {source}: {count} chunks")
        
        else:
            st.info("Loading index...")

    with st.expander("⚙️ Architecture Info"):

        st.write("**🧠 Model:** Mistral Large")
        st.write("**🔍 Search:** FAISS Vector DB")
        st.write("**📊 Embeddings:** Mistral Embed")

    with st.expander("💬 Question Examples", expanded=True):
        st.code("What was the Group's net income expected for 2025?", language=None)
        st.code("What was the interim dividend per share announced for 2025, and when will it be paid?", language=None)
        st.code("Summarize the bank's 2026 growth trajectory as outlined in the reports.", language=None)



# Complete RAG Pipeline with Citations - Reusable Function
def rag_with_citations(question, top_k=5):
    """
    Complete RAG pipeline with source citations

    Args:
        question (str): User's question
        top_k (int): Number of chunks to retrieve

    Returns:
        dict: Contains answer, sources, and metadata
    """

    # Get embeddings for the question
    question_embeddings = get_embeddings_with_retry([question], batch_size=1)
    question_embedding = np.array([question_embeddings[0]])

    # Search for most similar chunks
    distances, chunk_indices = st.session_state.similarity_index.search(question_embedding, k=top_k)
    relevant_chunks = [st.session_state.chunks[idx] for idx in chunk_indices[0]]

    # Format chunks with source information
    formatted_chunks = [
        format_chunk_with_source(chunk, i) for i, chunk in enumerate(relevant_chunks)
    ]

    # Create enhanced prompt
    enhanced_prompt = f"""You are an AI Assistant specialised in Financial Analysis. You are given a question and context information with sources. You need to answer the question based on the context and provide proper source citations.

Context information with sources is below:
---------------------
{"".join(formatted_chunks)}
---------------------

INSTRUCTIONS:
1. Answer the query based ONLY on the provided context information
2. For each fact or figure you mention, include a citation in the format [Source: Document Name (Page X)]
3. If information comes from multiple sources, cite all relevant sources
4. Be specific about which information comes from which source
5. If you cannot find information in the provided context, state that clearly

Query: {question}

Answer with proper source citations:
"""

    # Get answer with citations
    answer = run_mistral(enhanced_prompt)

    # Create source summary
    source_summary = create_source_summary(relevant_chunks)

    return {
        "question": question,
        "answer": answer,
        "source_summary": source_summary,
        "chunks_used": len(relevant_chunks),
        "sources": list(
            set(chunk.metadata.get("source", "Unknown") for chunk in relevant_chunks)
        ),
    }


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    avatar = ASSET_AVATAR if message["role"] == "assistant" else None
    with st.chat_message(message["role"], avatar=avatar):
        if message["role"] == "assistant" and "answer" in message:
            st.markdown(message["answer"])
            if "source_summary" in message:
                with st.expander(
                    f"📚 Sources ({message.get('chunks_used', 'N/A')} chunks)",
                    expanded=False,
                ):
                    st.markdown(message["source_summary"])
        else:
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input(
    "Ask a question about BNP Paribas Q2 2025 financial results..."
):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant", avatar=ASSET_AVATAR):
        with st.spinner("🔍 Searching through financial documents..."):
            try:
                result = rag_with_citations(prompt, top_k=5)

                # Display answer and sources
                st.markdown(result["answer"])
                with st.expander(
                    f"📚 Sources ({result['chunks_used']} chunks)", expanded=False
                ):
                    st.markdown(result["source_summary"])

                assistant_message = {
                    "role": "assistant",
                    "answer": result["answer"],
                    "source_summary": result["source_summary"],
                    "chunks_used": result["chunks_used"],
                }

            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                assistant_message = {"role": "assistant", "content": error_msg}

    # Add assistant response to chat history
    st.session_state.messages.append(assistant_message)
