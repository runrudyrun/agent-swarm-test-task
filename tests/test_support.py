"""Tests for Support Agent."""

import json
import tempfile
from pathlib import Path

import pytest

from agents.support_agent import SupportAgent
from tools.user_store import get_account_details, get_recent_transactions, open_support_ticket


class TestSupportAgent:
    """Test cases for Support Agent."""
    
    def test_support_agent_initialization(self):
        """Test support agent initialization."""
        agent = SupportAgent()
        
        assert agent is not None
        assert len(agent.get_tools()) == 3
        assert "get_account_details" in [tool.name for tool in agent.get_tools()]
        assert "get_recent_transactions" in [tool.name for tool in agent.get_tools()]
        assert "open_support_ticket" in [tool.name for tool in agent.get_tools()]
    
    def test_get_account_details_existing_user(self, temp_mock_data):
        """Test getting account details for existing user."""
        # Temporarily replace the data path
        original_path = "data/mock/users.json"
        
        try:
            # Use the temp file
            result = get_account_details("test_user_123")
            
            assert "Test User" in result
            assert "R$ 1.500,00" in result  # Check balance formatting
            assert "✅" in result  # Status emoji
            assert "Empresarial" in result  # Account type
        finally:
            pass  # Restore original behavior
    
    def test_get_account_details_non_existing_user(self):
        """Test getting account details for non-existing user."""
        result = get_account_details("non_existing_user")
        
        assert "❌" in result
        assert "não encontrado" in result.lower()
    
    def test_get_recent_transactions(self, temp_mock_data):
        """Test getting recent transactions."""
        result = get_recent_transactions("test_user_123", limit=5)
        
        assert "Test payment" in result
        assert "R$ 200,00" in result
        assert "✅" in result  # Status emoji
    
    def test_get_recent_transactions_no_transactions(self):
        """Test getting transactions for user with no transactions."""
        result = get_recent_transactions("non_existing_user")
        
        assert "não encontrado" in result.lower()
    
    def test_open_support_ticket(self, temp_mock_data):
        """Test opening a support ticket."""
        result = open_support_ticket(
            "test_user_123",
            "Test Subject",
            "Test description of the issue"
        )
        
        assert "✅" in result
        assert "Ticket Criado" in result
        assert "Test Subject" in result
        assert "ticket" in result.lower()
    
    def test_open_support_ticket_non_existing_user(self):
        """Test opening ticket for non-existing user."""
        result = open_support_ticket(
            "non_existing_user",
            "Test Subject",
            "Test description"
        )
        
        assert "❌" in result
        assert "não encontrado" in result.lower()
    
    def test_support_agent_balance_query(self):
        """Test support agent processing balance query."""
        agent = SupportAgent()
        
        result = agent.process_query("Qual é o meu saldo?", user_id="test_user_123")
        
        assert result["agent_used"] == "support"
        assert result["requires_user_id"] is False
        # Note: The actual content will depend on the mock data availability
    
    def test_support_agent_transactions_query(self):
        """Test support agent processing transactions query."""
        agent = SupportAgent()
        
        result = agent.process_query("Quero ver minhas transações", user_id="test_user_123")
        
        assert result["agent_used"] == "support"
        assert result["requires_user_id"] is False
    
    def test_support_agent_support_ticket_query(self):
        """Test support agent processing support ticket query."""
        agent = SupportAgent()
        
        result = agent.process_query("Preciso de ajuda com um problema", user_id="test_user_123")
        
        assert result["agent_used"] == "support"
        # Should ask for more details about the problem
    
    def test_support_agent_requires_user_id(self):
        """Test support agent requiring user ID."""
        agent = SupportAgent()
        
        result = agent.process_query("Qual é o meu saldo?")  # No user_id provided
        
        assert result["requires_user_id"] is True
        assert "ID de usuário" in result["answer"]
    
    def test_tool_suggestions(self):
        """Test tool suggestion functionality."""
        from tools.user_store import get_tool_suggestions
        
        # Test balance-related query
        suggestions = get_tool_suggestions("Qual é o meu saldo?")
        assert "get_account_details" in suggestions
        
        # Test transactions-related query
        suggestions = get_tool_suggestions("Quero ver minhas transações")
        assert "get_recent_transactions" in suggestions
        
        # Test support-related query
        suggestions = get_tool_suggestions("Preciso de ajuda")
        assert "open_support_ticket" in suggestions
    
    def test_extract_limit(self):
        """Test limit extraction from query."""
        agent = SupportAgent()
        
        assert agent._extract_limit("Mostre 10 transações") == 10
        assert agent._extract_limit("Quero ver 3 movimentações") == 3
        assert agent._extract_limit("Mostre transações") == 5  # Default
        assert agent._extract_limit("Mostre 100 transações") == 50  # Max limit
    
    def test_extract_ticket_info(self):
        """Test ticket info extraction."""
        agent = SupportAgent()
        
        subject, description = agent._extract_ticket_info(
            "Estou com problema na maquininha. Ela não conecta no Wi-Fi e já tentei reiniciar várias vezes."
        )
        
        assert subject is not None
        assert description is not None
        assert "problema na maquininha" in subject.lower()
    
    def test_support_agent_default_response(self):
        """Test support agent default response for unclear queries."""
        agent = SupportAgent()
        
        result = agent.process_query("asdfgh", user_id="test_user_123")
        
        assert "Posso ajudar você com" in result["answer"]
        assert "Dados da conta" in result["answer"]
        assert "Transações" in result["answer"]
        assert "Suporte" in result["answer"]