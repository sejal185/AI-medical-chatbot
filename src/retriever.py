"""
Vectorstore Retriever:
Loads the persisted FAISS vector index and retrieves the most relevant
document chunks for a user's medical query.
"""

from typing import List, Tuple
from langchain_core.documents import Document

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS

try:
    from src.config import (
        FAISS_INDEX_PATH,
        EMBEDDING_MODEL_NAME,
        EMBEDDING_DEVICE,
        TOP_K_RESULTS,
    )
except ImportError:
    from config import (
        FAISS_INDEX_PATH,
        EMBEDDING_MODEL_NAME,
        EMBEDDING_DEVICE,
        TOP_K_RESULTS,
    )


def is_vectorstore_available() -> bool:
    """
    Checks whether the FAISS vector database files exist locally.
    """
    index_file = FAISS_INDEX_PATH / "index.faiss"
    pkl_file = FAISS_INDEX_PATH / "index.pkl"
    return index_file.exists() and pkl_file.exists()


class MedicalRetriever:
    """
    Handles FAISS vectorstore loading and document similarity retrieval.
    """

    def __init__(self, top_k: int = TOP_K_RESULTS):
        self.top_k = top_k
        self._vectorstore = None
        self._embeddings = None

    def _get_embeddings(self) -> HuggingFaceEmbeddings:
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={"device": EMBEDDING_DEVICE},
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._embeddings

    def load_vectorstore(self) -> FAISS:
        """
        Loads the FAISS index from local disk.
        """
        if not is_vectorstore_available():
            raise FileNotFoundError(
                f"FAISS index not found at '{FAISS_INDEX_PATH}'. "
                "Please run data ingestion first using `python src/ingest.py`."
            )

        if self._vectorstore is None:
            embeddings = self._get_embeddings()
            self._vectorstore = FAISS.load_local(
                folder_path=str(FAISS_INDEX_PATH),
                embeddings=embeddings,
                allow_dangerous_deserialization=True,
            )
        return self._vectorstore

    def retrieve(self, query: str, top_k: int = None) -> List[Document]:
        """
        Performs similarity search to find the most relevant document chunks.
        """
        k = top_k or self.top_k
        vectorstore = self.load_vectorstore()
        return vectorstore.similarity_search(query, k=k)

    def retrieve_with_scores(
        self, query: str, top_k: int = None
    ) -> List[Tuple[Document, float]]:
        """
        Performs similarity search with relevance distance/scores.
        """
        k = top_k or self.top_k
        vectorstore = self.load_vectorstore()
        return vectorstore.similarity_search_with_score(query, k=k)

    @staticmethod
    def format_context(documents: List[Document]) -> str:
        """
        Formats retrieved documents into a clean string for the LLM prompt.
        """
        from pathlib import Path
        formatted_chunks = []
        for i, doc in enumerate(documents, start=1):
            raw_source = doc.metadata.get("source", "Medical Reference Document")
            source = Path(raw_source).name
            page = doc.metadata.get("page", "N/A")
            header = f"--- [Document Chunk {i} | Source: {source} | Page: {page}] ---"
            formatted_chunks.append(f"{header}\n{doc.page_content.strip()}")

        return "\n\n".join(formatted_chunks)
