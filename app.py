"""
Medical RAG Chatbot - Streamlit Application
A clean, grounded RAG chatbot powered by LangChain, HuggingFace Embeddings,
FAISS, and an LLM API.
"""

import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

# Ensure environment variables from .env are loaded
load_dotenv()

from src.config import (
    get_pdf_path,
    MEDICAL_DISCLAIMER,
    FALLBACK_MESSAGE,
    DATA_DIR,
    FAISS_INDEX_PATH,
)
from src.retriever import MedicalRetriever, is_vectorstore_available
from src.llm import get_llm, generate_medical_answer
from src.ingest import run_ingestion

# Streamlit Page Setup
st.set_page_config(
    page_title="Medical RAG Chatbot",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for polished portfolio presentation
st.markdown(
    """
    <style>
    .medical-disclaimer-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 12px 18px;
        border-radius: 8px;
        border-left: 5px solid #ffeeba;
        margin-bottom: 20px;
        font-size: 0.95rem;
    }
    .chunk-source-tag {
        background-color: #e9ecef;
        color: #495057;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 6px;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Detect Document Automatically
detected_pdf_name = None
try:
    pdf_path = get_pdf_path()
    detected_pdf_name = pdf_path.name
except Exception:
    detected_pdf_name = None

@st.cache_resource
def get_cached_retriever(top_k: int) -> MedicalRetriever:
    return MedicalRetriever(top_k=top_k)

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Configuration")

    # Document Information
    st.subheader("📄 Reference Document")
    if detected_pdf_name:
        st.success(f"**Detected File:**\n`{detected_pdf_name}`")
    else:
        st.error(f"No PDF detected in `{DATA_DIR}` folder.")

    # Vectorstore Status
    st.subheader("🧠 Vector Database")
    db_ready = is_vectorstore_available()
    if db_ready:
        st.success("✅ FAISS Index Ready")
    else:
        st.warning("⚠️ FAISS Index Not Found")

    # Ingestion Trigger Button
    if st.button("⚡ Generate / Rebuild FAISS Index", use_container_width=True):
        if not detected_pdf_name:
            st.error("Cannot index: No PDF document found in data/ directory.")
        else:
            with st.spinner("Processing PDF, generating embeddings, and building FAISS vectorstore..."):
                try:
                    num_pages, num_chunks = run_ingestion()
                    st.cache_resource.clear()
                    st.success(f"Index built! Processed {num_pages} pages into {num_chunks} chunks.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

    st.divider()

    # LLM Settings
    st.subheader("🤖 LLM Model & Provider")
    env_provider = os.getenv("LLM_PROVIDER", "groq").lower()
    provider_options = ["Groq", "Google Gemini", "OpenAI"]
    default_idx = 0
    if "gemini" in env_provider or "google" in env_provider:
        default_idx = 1
    elif "openai" in env_provider:
        default_idx = 2

    provider_choice = st.selectbox(
        "Select Provider",
        options=provider_options,
        index=default_idx,
    )
    provider_map = {
        "Groq": "groq",
        "Google Gemini": "google",
        "OpenAI": "openai",
    }
    selected_provider = provider_map[provider_choice]

    # Automatic Environment Key Detection (Do NOT display the actual key)
    api_key_input = None
    has_env_key = False
    if selected_provider == "groq":
        has_env_key = bool(os.getenv("GROQ_API_KEY"))
    elif selected_provider == "google":
        has_env_key = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
    elif selected_provider == "openai":
        has_env_key = bool(os.getenv("OPENAI_API_KEY"))

    if has_env_key:
        st.success(f"🔒 {provider_choice} API Key loaded from `.env`")
    else:
        st.warning(f"⚠️ No API key found in `.env` for {provider_choice}")
        api_key_input = st.text_input(
            f"Enter {provider_choice} API Key",
            type="password",
            help="Alternatively, configure this key in your .env file.",
        )

    # Retrieval parameter
    top_k = st.slider("Retrieved Chunks (k)", min_value=1, max_value=8, value=4)

    st.divider()

    # Reset Chat
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# --- MAIN CHAT INTERFACE ---
st.title("🩺 Medical RAG Chatbot")
st.caption("Grounded retrieval-augmented generation from verified medical reference literature.")

# Visible Medical Disclaimer (Required)
st.markdown(
    f"""
    <div class="medical-disclaimer-box">
        <strong>⚠️ Medical Disclaimer:</strong> {MEDICAL_DISCLAIMER}
    </div>
    """,
    unsafe_allow_html=True,
)

# Display Existing Chat Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If there are sources attached, render them
        if "sources" in message and message["sources"]:
            with st.expander("📚 Retrieved Reference Context & Sources"):
                for idx, src in enumerate(message["sources"], start=1):
                    source_name = src.get("source", "Unknown Document")
                    page_num = src.get("page", "N/A")
                    st.markdown(
                        f"<div class='chunk-source-tag'>Chunk {idx} | Document: {source_name} | Page {page_num}</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"> *{src.get('text', '').strip()}*")

# Chat Input Handler
if prompt := st.chat_input("Ask a medical question based on the document..."):
    # Append user question
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Ensure vectorstore is available
    if not is_vectorstore_available():
        with st.chat_message("assistant"):
            warning_msg = (
                "⚠️ **FAISS Vector Database is not generated yet!**\n\n"
                "Please click **'Generate / Rebuild FAISS Index'** in the sidebar, or run:\n"
                "```bash\npython src/ingest.py\n```"
            )
            st.markdown(warning_msg)
            st.session_state.messages.append({"role": "assistant", "content": warning_msg})
    else:
        with st.chat_message("assistant"):
            with st.spinner("Retrieving medical context and generating answer..."):
                try:
                    # 1. Retrieve most relevant chunks
                    retriever = get_cached_retriever(top_k=top_k)
                    retrieved_docs = retriever.retrieve(prompt)

                    # 2. Format context
                    context = MedicalRetriever.format_context(retrieved_docs)

                    # 3. Generate grounded answer
                    answer = generate_medical_answer(
                        question=prompt,
                        context=context,
                        provider=selected_provider,
                        api_key=api_key_input if api_key_input else None,
                    )

                    st.markdown(answer)

                    # Prepare sources for display and persistence
                    sources_data = [
                        {
                            "source": Path(doc.metadata.get("source", "Reference Document")).name,
                            "page": doc.metadata.get("page", "N/A"),
                            "text": doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""),
                        }
                        for doc in retrieved_docs
                    ]

                    # 4. Display source context
                    with st.expander("📚 Retrieved Reference Context & Sources"):
                        for idx, src in enumerate(sources_data, start=1):
                            st.markdown(
                                f"<div class='chunk-source-tag'>Chunk {idx} | Document: {src['source']} | Page {src['page']}</div>",
                                unsafe_allow_html=True,
                            )
                            st.markdown(f"> *{src['text']}*")

                    # Append to session state
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources_data,
                        }
                    )

                except Exception as err:
                    err_msg = f"❌ **Error generating response:** {err}"
                    st.error(err_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": err_msg}
                    )
