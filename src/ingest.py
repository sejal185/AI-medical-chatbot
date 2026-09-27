"""
Data Ingestion Pipeline:
1. Automatically inspects data/ for the medical PDF.
2. Extracts text using PyPDF.
3. Splits documents into manageable chunks.
4. Generates embeddings with HuggingFace sentence-transformers.
5. Builds and saves a local FAISS vector store.
"""

import sys
from pathlib import Path
from typing import Tuple, List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Handle HuggingFaceEmbeddings import cleanly across LangChain versions
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS

# Support running directly or as a module
try:
    from src.config import (
        get_pdf_path,
        VECTORSTORE_DIR,
        FAISS_INDEX_PATH,
        EMBEDDING_MODEL_NAME,
        EMBEDDING_DEVICE,
        CHUNK_SIZE,
        CHUNK_OVERLAP,
    )
except ImportError:
    from config import (
        get_pdf_path,
        VECTORSTORE_DIR,
        FAISS_INDEX_PATH,
        EMBEDDING_MODEL_NAME,
        EMBEDDING_DEVICE,
        CHUNK_SIZE,
        CHUNK_OVERLAP,
    )


def load_pdf_documents(pdf_path: Path) -> List[Document]:
    """
    Extracts text and page metadata from the PDF using PyPDF.
    """
    print(f"[1/4] Loading PDF: {pdf_path.name}")
    try:
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
    except Exception as e:
        print(f"      PyPDFLoader encountered an issue ({e}). Falling back to pypdf.PdfReader...")
        import pypdf
        reader = pypdf.PdfReader(str(pdf_path))
        docs = []
        for idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": pdf_path.name, "page": idx + 1},
                    )
                )

    print(f"      Extracted {len(docs)} pages with content.")
    return docs


def split_documents(docs: List[Document]) -> List[Document]:
    """
    Splits loaded documents into chunks with overlap for contextual integrity.
    """
    print(f"[2/4] Splitting text into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = text_splitter.split_documents(docs)
    print(f"      Generated {len(chunks)} text chunks.")
    return chunks


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Initializes HuggingFace sentence-transformers embeddings.
    """
    print(f"[3/4] Initializing embedding model: {EMBEDDING_MODEL_NAME} ({EMBEDDING_DEVICE})...")
    model_kwargs = {"device": EMBEDDING_DEVICE}
    encode_kwargs = {"normalize_embeddings": True}
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    return embeddings


def build_and_save_vectorstore(
    chunks: List[Document], embeddings: HuggingFaceEmbeddings
) -> FAISS:
    """
    Builds FAISS index from chunks and persists it locally.
    """
    print("[4/4] Generating embeddings and building FAISS vector database...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Ensure output directory exists
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(FAISS_INDEX_PATH))
    print(f"      FAISS index successfully saved to: {FAISS_INDEX_PATH}")
    return vectorstore


def run_ingestion(max_pages: int = None) -> Tuple[int, int]:
    """
    Full pipeline to ingest the detected PDF and build the vectorstore.
    Optional max_pages can be used to limit initial indexing for large encyclopedias.
    """
    pdf_path = get_pdf_path()
    print(f"\n==========================================")
    print(f"Medical RAG Ingestion Pipeline")
    print(f"Detected document: {pdf_path}")
    print(f"==========================================")

    docs = load_pdf_documents(pdf_path)
    if max_pages and max_pages < len(docs):
        print(f"Note: Limiting to first {max_pages} pages for fast indexing.")
        docs = docs[:max_pages]

    chunks = split_documents(docs)
    embeddings = get_embedding_model()
    build_and_save_vectorstore(chunks, embeddings)

    print(f"\nIngestion Complete!")
    print(f"Total Pages Processed: {len(docs)}")
    print(f"Total Chunks Stored: {len(chunks)}")
    return len(docs), len(chunks)


if __name__ == "__main__":
    max_pages = None
    if len(sys.argv) > 1:
        try:
            max_pages = int(sys.argv[1])
        except ValueError:
            pass
    run_ingestion(max_pages=max_pages)
