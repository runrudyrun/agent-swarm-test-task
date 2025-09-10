"""Test configuration and fixtures."""

import json
import os
import tempfile
from pathlib import Path
from typing import Dict

import pytest
from fastapi.testclient import TestClient

# Set test environment variables before imports
os.environ["ENVIRONMENT"] = "test"
os.environ["PERSONALITY"] = "off"  # Disable personality for deterministic tests

from api.main import app
from agents.router_agent import RouterAgent


@pytest.fixture
def test_client():
    """Create FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def mock_user_data():
    """Mock user data for testing."""
    return {
        "users": [
            {
                "id": "test_user_123",
                "name": "Test User",
                "email": "test@example.com",
                "phone": "+55 11 98765-4321",
                "account_type": "business",
                "balance": 1500.00,
                "status": "active",
                "created_at": "2023-01-01T10:00:00Z"
            }
        ],
        "transactions": [
            {
                "id": "txn_test_001",
                "user_id": "test_user_123",
                "type": "payment",
                "amount": 200.00,
                "description": "Test payment",
                "status": "completed",
                "created_at": "2024-01-15T15:30:00Z"
            }
        ],
        "support_tickets": []
    }


@pytest.fixture
def temp_mock_data(mock_user_data):
    """Create temporary mock data file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(mock_user_data, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    os.unlink(temp_path)


@pytest.fixture
def router_agent():
    """Create router agent instance for testing."""
    return RouterAgent()


@pytest.fixture
def sample_queries():
    """Sample queries for testing."""
    return {
        "support": [
            "Qual é o meu saldo?",
            "Quero ver minhas transações recentes",
            "Estou com problema na minha conta",
            "Preciso de ajuda com minha maquininha"
        ],
        "knowledge": [
            "O que é o InfinitePay?",
            "Como funciona a maquininha?",
            "Quais são as taxas?",
            "Como faço para abrir uma conta?"
        ],
        "multi_intent": [
            "Quero saber meu saldo e também como funciona a maquininha",
            "Qual é o meu extrato e o que é o InfinitePay?"
        ],
        "escalation": [
            "Preciso falar com um atendente humano",
            "É urgente, me ajude imediatamente",
            "Não estou entendendo nada"
        ]
    }


@pytest.fixture
def mock_vector_store():
    """Create mock vector store for testing."""
    # Create a temporary directory for mock vector store
    temp_dir = tempfile.mkdtemp()
    
    # Create mock documents
    from langchain.schema import Document
    
    mock_docs = [
        Document(
            page_content="A InfinitePay oferece maquininhas de cartão com taxas competitivas. Nossos produtos incluem soluções para pequenas e médias empresas.",
            metadata={"source": "https://www.infinitepay.io/maquininha", "title": "Maquininhas de Cartão"}
        ),
        Document(
            page_content="Para abrir uma conta na InfinitePay, você precisa de CPF, comprovante de residência e documento de identidade. O processo é simples e rápido.",
            metadata={"source": "https://www.infinitepay.io/conta-digital", "title": "Abertura de Conta"}
        )
    ]
    
    yield mock_docs, temp_dir
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)