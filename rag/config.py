"""RAG configuration and settings."""

import os
from typing import List

from langchain.embeddings.base import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from chromadb.config import Settings


def get_llm_config():
    """Get LLM configuration based on environment variables."""
    import os
    
    provider = os.getenv("MODEL_PROVIDER", "local")
    
    if provider == "openai" and os.getenv("OPENAI_API_KEY"):
        model = os.getenv("OPENAI_MODEL", "gpt-5-mini-2025-08-07")  # Default to GPT-5 mini
        return {
            "provider": "openai",
            "model": model,
            "temperature": 0.1,
            "api_key": os.getenv("OPENAI_API_KEY")
        }
    else:
        return {
            "provider": "local",
            "model": "mock"
        }


def create_llm():
    """Create LLM instance based on configuration."""
    config = get_llm_config()
    
    if config["provider"] == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            temperature=config["temperature"],
            model=config["model"],
            api_key=config["api_key"]
        )
    else:
        # Return MockLLM for local development
        from langchain.llms.base import LLM
        
        class MockLLM(LLM):
            def _call(self, prompt, stop=None):
                # Simple mock that extracts relevant info from context
                if "CONTEXTO:" in prompt:
                    context = prompt.split("CONTEXTO:")[1].split("PERGUNTA:")[0].strip()
                    question = prompt.split("PERGUNTA:")[1].split("INSTRUÇÕES:")[0].strip()
                    
                    # Simple response based on context
                    if "taxa" in question.lower() or "preço" in question.lower():
                        return "Para informações atualizadas sobre taxas, recomendo consultar nosso site oficial ou entrar em contato com o suporte."
                    elif "funciona" in question.lower() or "como usar" in question.lower():
                        return "Com base nas informações disponíveis, posso explicar como funcionam nossos produtos. Para detalhes específicos, consulte nosso site."
                    else:
                        return "Com base nas informações disponíveis em nosso banco de dados, posso ajudar com informações gerais sobre produtos e serviços InfinitePay."
                
                return "Desculpe, não consegui processar sua pergunta."
            
            def invoke(self, prompt, **kwargs):
                # Return an object with content attribute to match OpenAI response format
                response_text = self._call(prompt)
                class MockResponse:
                    def __init__(self, content):
                        self.content = content
                return MockResponse(response_text)
            
            @property
            def _llm_type(self):
                return "mock"
        
        return MockLLM()


class RAGConfig:
    """Configuration for RAG system."""
    
    # Vector store settings
    VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./data/chroma")
    COLLECTION_NAME = "infinitepay_knowledge"
    
    # Embedding settings
    LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
    
    # Chunking settings
    CHUNK_SIZE = 800  # tokens
    CHUNK_OVERLAP = 100  # tokens
    
    # Retrieval settings
    TOP_K = 5
    MMR_K = 5
    MMR_FETCH_K = 20
    
    # InfinitePay URLs to scrape
    INFINITEPAY_URLS = [
        "https://www.infinitepay.io",
        "https://www.infinitepay.io/maquininha",
        "https://www.infinitepay.io/maquininha-celular",
        "https://www.infinitepay.io/tap-to-pay",
        "https://www.infinitepay.io/pdv",
        "https://www.infinitepay.io/receba-na-hora",
        "https://www.infinitepay.io/gestao-de-cobranca-2",
        "https://www.infinitepay.io/gestao-de-cobranca",
        "https://www.infinitepay.io/link-de-pagamento",
        "https://www.infinitepay.io/loja-online",
        "https://www.infinitepay.io/boleto",
        "https://www.infinitepay.io/conta-digital",
        "https://www.infinitepay.io/conta-pj",
        "https://www.infinitepay.io/pix",
        "https://www.infinitepay.io/pix-parcelado",
        "https://www.infinitepay.io/emprestimo",
        "https://www.infinitepay.io/cartao",
        "https://www.infinitepay.io/rendimento",
    ]


def get_embeddings() -> Embeddings:
    """Get embeddings based on configuration."""
    provider = os.getenv("EMBEDDINGS_PROVIDER", "local")
    
    if provider == "openai" and os.getenv("OPENAI_API_KEY"):
        return OpenAIEmbeddings(model=RAGConfig.OPENAI_EMBEDDING_MODEL)
    else:
        return HuggingFaceEmbeddings(model_name=RAGConfig.LOCAL_EMBEDDING_MODEL)


def get_chroma_settings() -> Settings:
    """Return explicit Chroma client settings to avoid deprecated configuration and control backend.

    Honors env vars:
    - CHROMA_DB_IMPL (default: duckdb+parquet)
    - VECTOR_STORE_PATH (default: ./data/chroma)
    - CHROMA_ANONYMIZED_TELEMETRY (default: false)
    - CHROMA_ALLOW_RESET (default: false)
    """
    impl = os.getenv("CHROMA_DB_IMPL", "duckdb+parquet")
    allow_reset = os.getenv("CHROMA_ALLOW_RESET", "false").lower() in {"1", "true", "yes"}
    telemetry = os.getenv("CHROMA_ANONYMIZED_TELEMETRY", "false").lower() in {"1", "true", "yes"}

    return Settings(
        chroma_db_impl=impl,
        persist_directory=RAGConfig.VECTOR_STORE_PATH,
        allow_reset=allow_reset,
        anonymized_telemetry=telemetry,
    )