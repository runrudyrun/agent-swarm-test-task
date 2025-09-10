"""RAG configuration and settings."""

import os
from typing import List

from langchain.embeddings.base import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings


class RAGConfig:
    """Configuration for RAG system."""
    
    # Vector store settings
    VECTOR_STORE_PATH = "./data/chroma"
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