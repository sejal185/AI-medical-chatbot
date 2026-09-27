import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
FAISS_INDEX_PATH = VECTORSTORE_DIR / "faiss_index"

# Embedding Model Configuration
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

# Text Splitter Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# Retrieval Configuration
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "4"))

# Mandatory Disclaimers and Fallback Responses
MEDICAL_DISCLAIMER = (
    "This chatbot provides informational content based on the provided reference "
    "document and is not a substitute for professional medical advice, diagnosis, or treatment."
)

FALLBACK_MESSAGE = (
    "I could not find sufficient information about this in the provided medical document."
)

# Strict Grounded RAG Prompt Template
RAG_SYSTEM_PROMPT = f"""You are a specialized medical information assistant.
Answer the user's medical question using ONLY the provided reference document context below.

STRICT GUIDELINES:
1. Ground your answer completely in the retrieved context.
2. If the retrieved context does not contain sufficient facts to answer the question accurately and safely, respond ONLY with:
   "{FALLBACK_MESSAGE}"
3. Do NOT invent, assume, or extrapolate any medical information beyond the context.
4. Keep the response factual, concise, and easy to understand.

Reference Context:
{{context}}

Question:
{{question}}

Answer:"""


def get_pdf_path() -> Path:
    """
    Automatically inspects the data directory to locate the medical PDF document.
    Does not assume a hardcoded filename.
    """
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found at: {DATA_DIR}")

    pdf_files = list(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF document found in '{DATA_DIR}'. Please place your medical PDF in the data/ folder."
        )

    # Return the first PDF found
    return pdf_files[0]
