"""
LLM Integration Module:
Initializes the Language Model and executes the grounded RAG chain.
Supports multiple LLM providers:
- Google Gemini (Gemini 1.5 Flash / Gemini 2.0 Flash)
- Groq (Llama 3.3 / Llama 3)
- OpenAI (GPT-4o mini)
"""

import os
from typing import Optional
from langchain_core.messages import SystemMessage, HumanMessage

try:
    from src.config import RAG_SYSTEM_PROMPT, FALLBACK_MESSAGE, MEDICAL_DISCLAIMER
except ImportError:
    from config import RAG_SYSTEM_PROMPT, FALLBACK_MESSAGE, MEDICAL_DISCLAIMER


def get_llm(
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.1,
):
    """
    Initializes and returns a configured LangChain Chat model.
    Prioritizes explicit provider/key, otherwise inspects environment variables.
    """
    # Detect provider based on available environment variables or parameter
    detected_provider = provider or os.getenv("LLM_PROVIDER")

    google_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    groq_key = api_key or os.getenv("GROQ_API_KEY")
    openai_key = api_key or os.getenv("OPENAI_API_KEY")

    if not detected_provider:
        if google_key:
            detected_provider = "google"
        elif groq_key:
            detected_provider = "groq"
        elif openai_key:
            detected_provider = "openai"
        else:
            detected_provider = "google"  # default fallback

    detected_provider = detected_provider.lower()

    if detected_provider in ["google", "gemini"]:
        effective_key = google_key if (api_key or google_key) else None
        if not effective_key:
            raise ValueError(
                "Google Gemini API key not found. Please provide GOOGLE_API_KEY in your .env file or Streamlit sidebar."
            )
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=effective_key,
                temperature=temperature,
            )
        except ImportError:
            raise ImportError(
                "langchain-google-genai is required for Google Gemini. Run: pip install langchain-google-genai"
            )

    elif detected_provider == "groq":
        effective_key = groq_key if (api_key or groq_key) else None
        if not effective_key:
            raise ValueError(
                "Groq API key not found. Please provide GROQ_API_KEY in your .env file or Streamlit sidebar."
            )
        try:
            from langchain_groq import ChatGroq
            model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
            return ChatGroq(
                model_name=model_name,
                groq_api_key=effective_key,
                temperature=temperature,
            )
        except ImportError:
            # Fallback to langchain_openai with Groq base URL if langchain_groq isn't installed
            try:
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                    api_key=effective_key,
                    base_url="https://api.groq.com/openai/v1",
                    temperature=temperature,
                )
            except ImportError:
                raise ImportError(
                    "Either langchain-groq or langchain-openai is required for Groq. Run: pip install langchain-groq"
                )

    elif detected_provider == "openai":
        effective_key = openai_key if (api_key or openai_key) else None
        if not effective_key:
            raise ValueError(
                "OpenAI API key not found. Please provide OPENAI_API_KEY in your .env file or Streamlit sidebar."
            )
        try:
            from langchain_openai import ChatOpenAI
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            return ChatOpenAI(
                model=model_name,
                api_key=effective_key,
                temperature=temperature,
            )
        except ImportError:
            raise ImportError(
                "langchain-openai is required for OpenAI. Run: pip install langchain-openai"
            )

    else:
        raise ValueError(
            f"Unsupported LLM provider '{detected_provider}'. Supported options: 'google', 'groq', 'openai'."
        )


def generate_medical_answer(
    question: str,
    context: str,
    llm=None,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """
    Generates a strictly grounded medical answer from the retrieved document context.
    """
    if not context or not context.strip():
        return FALLBACK_MESSAGE

    if llm is None:
        llm = get_llm(provider=provider, api_key=api_key)

    prompt_text = RAG_SYSTEM_PROMPT.format(context=context, question=question)

    messages = [
        SystemMessage(content="You are a medical RAG assistant that answers questions solely based on provided documents."),
        HumanMessage(content=prompt_text),
    ]

    response = llm.invoke(messages)
    answer = response.content.strip() if hasattr(response, "content") else str(response).strip()
    return answer
