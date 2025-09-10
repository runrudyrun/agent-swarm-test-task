"""Tests for Router Agent."""

import pytest

from agents.router_agent import RouterAgent


class TestRouterAgent:
    """Test cases for Router Agent."""
    
    def test_classify_support_intent(self, router_agent):
        """Test classification of support intents."""
        test_cases = [
            ("Qual é o meu saldo?", "support"),
            ("Quero ver minhas transações", "support"),
            ("Estou com problema na minha conta", "support"),
            ("Meu ID é user123 e quero meu extrato", "support"),
        ]
        
        for query, expected_intent in test_cases:
            intent, confidence = router_agent.classify_intent(query)
            assert intent == expected_intent
            assert confidence > 0.5
    
    def test_classify_knowledge_intent(self, router_agent):
        """Test classification of knowledge intents."""
        test_cases = [
            ("O que é o InfinitePay?", "knowledge"),
            ("Como funciona a maquininha?", "knowledge"),
            ("Quais são as taxas?", "knowledge"),
            ("Como faço para abrir uma conta?", "knowledge"),
        ]
        
        for query, expected_intent in test_cases:
            intent, confidence = router_agent.classify_intent(query)
            assert intent == "knowledge"
            assert confidence > 0.5
    
    def test_classify_escalation_intent(self, router_agent):
        """Test classification of escalation intents."""
        test_cases = [
            ("Preciso falar com um atendente humano", "escalate"),
            ("É urgente, me ajude imediatamente", "escalate"),
            ("Não estou entendendo nada", "escalate"),
        ]
        
        for query, expected_intent in test_cases:
            intent, confidence = router_agent.classify_intent(query)
            assert intent == "escalate"
            assert confidence == 1.0
    
    def test_classify_unknown_intent(self, router_agent):
        """Test classification of unknown intents."""
        test_cases = [
            ("asdfgh", "unknown"),
            ("12345", "unknown"),
            ("", "unknown"),
        ]
        
        for query, expected_intent in test_cases:
            intent, confidence = router_agent.classify_intent(query)
            assert intent == expected_intent
            assert confidence < 0.5
    
    def test_multi_intent_detection(self, router_agent):
        """Test detection of multi-intent queries."""
        query = "Quero saber meu saldo e também como funciona a maquininha"
        sub_queries = router_agent._split_multi_intent(query)
        
        assert len(sub_queries) >= 2  # Should split into at least 2 parts
    
    def test_route_support_query(self, router_agent):
        """Test routing support queries."""
        query = "Qual é o meu saldo?"
        result = router_agent.route_query(query, user_id="test_user_123")
        
        assert result["agent_used"] == "support"
        assert "intent" in result
        assert "confidence" in result
    
    def test_route_knowledge_query(self, router_agent):
        """Test routing knowledge queries."""
        query = "O que é o InfinitePay?"
        result = router_agent.route_query(query)
        
        assert result["agent_used"] in ["knowledge", "router"]
        assert "intent" in result
        assert "confidence" in result
    
    def test_route_escalation_query(self, router_agent):
        """Test routing escalation queries."""
        query = "Preciso falar com um atendente humano"
        result = router_agent.route_query(query)
        
        assert result["handoff_to_human"] is True
        assert "atendente humano" in result["answer"].lower()
    
    def test_route_multi_intent_query(self, router_agent):
        """Test routing multi-intent queries."""
        query = "Quero saber meu saldo e também como funciona a maquininha"
        result = router_agent.route_query(query, user_id="test_user_123")
        
        assert result["intent"] == "multi_intent"
        assert "sub_queries" in result
        assert len(result["sub_queries"]) >= 2
    
    def test_escalation_patterns(self, router_agent):
        """Test escalation pattern detection."""
        escalation_queries = [
            "É urgente!",
            "Preciso falar com humano",
            "Não entendi nada",
            "Já tentei várias vezes",
        ]
        
        for query in escalation_queries:
            assert router_agent._check_escalation(query.lower()) is True
    
    def test_support_keywords(self, router_agent):
        """Test support keyword detection."""
        support_queries = [
            "minha conta",
            "meu saldo",
            "transação",
            "problema com",
        ]
        
        for query in support_queries:
            assert router_agent._has_support_keywords(query) is True
    
    def test_knowledge_keywords(self, router_agent):
        """Test knowledge keyword detection."""
        knowledge_queries = [
            "o que é",
            "como funciona",
            "taxa",
            "produto",
        ]
        
        for query in knowledge_queries:
            assert router_agent._has_knowledge_keywords(query) is True
    
    def test_agent_capabilities(self, router_agent):
        """Test agent capabilities retrieval."""
        capabilities = router_agent.get_agent_capabilities()
        
        assert "support" in capabilities
        assert "knowledge" in capabilities
        assert "description" in capabilities["support"]
        assert "capabilities" in capabilities["support"]
        assert "description" in capabilities["knowledge"]
        assert "capabilities" in capabilities["knowledge"]