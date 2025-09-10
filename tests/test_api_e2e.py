"""End-to-end tests for the FastAPI application."""

import json
import pytest
from datetime import datetime

from tests.conftest import test_client


class TestAPIE2E:
    """End-to-end test cases for the API."""
    
    def test_root_endpoint(self, test_client):
        """Test root endpoint."""
        response = test_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "InfinitePay Agent Swarm API" in data["message"]
        assert data["version"] == "0.1.0"
        assert "/docs" in data["docs"]
        assert "/health" in data["health"]
    
    def test_health_endpoint(self, test_client):
        """Test health check endpoint."""
        response = test_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] in ["healthy", "degraded"]
        assert data["timestamp"]
        assert data["version"] == "0.1.0"
        assert "agents" in data
        
        # Check individual agent statuses
        agents = data["agents"]
        assert "router" in agents
        assert "knowledge" in agents
        assert "support" in agents
        assert "personality" in agents
    
    def test_capabilities_endpoint(self, test_client):
        """Test capabilities endpoint."""
        response = test_client.get("/capabilities")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "agents" in data
        assert "features" in data
        assert "supported_locales" in data
        assert "max_query_length" in data
        
        # Check agent capabilities
        agents = data["agents"]
        assert "support" in agents
        assert "knowledge" in agents
        
        # Check features
        features = data["features"]
        assert "multi_intent" in features
        assert "personality_layer" in features
        assert "human_escalation" in features
        assert "source_citations" in features
    
    def test_query_endpoint_support(self, test_client):
        """Test query endpoint with support query."""
        query_data = {
            "message": "Qual é o meu saldo?",
            "user_id": "test_user_123"
        }
        
        response = test_client.post("/query", json=query_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert data["agent_used"] == "support"
        assert "intent" in data
        assert "confidence" in data
        assert isinstance(data["confidence"], float)
        assert data["requires_user_id"] is False
    
    def test_query_endpoint_knowledge(self, test_client):
        """Test query endpoint with knowledge query."""
        query_data = {
            "message": "O que é o InfinitePay?"
        }
        
        response = test_client.post("/query", json=query_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert data["agent_used"] in ["knowledge", "router"]
        assert "intent" in data
        assert "confidence" in data
    
    def test_query_endpoint_no_user_id_required(self, test_client):
        """Test query endpoint requiring user ID."""
        query_data = {
            "message": "Qual é o meu saldo?"
            # No user_id provided
        }
        
        response = test_client.post("/query", json=query_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["requires_user_id"] is True
        assert "ID de usuário" in data["answer"]
    
    def test_query_endpoint_escalation(self, test_client):
        """Test query endpoint with escalation query."""
        query_data = {
            "message": "Preciso falar com um atendente humano"
        }
        
        response = test_client.post("/query", json=query_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["handoff_to_human"] is True
        assert "atendente humano" in data["answer"].lower()
    
    def test_query_endpoint_multi_intent(self, test_client):
        """Test query endpoint with multi-intent query."""
        query_data = {
            "message": "Quero saber meu saldo e também como funciona a maquininha",
            "user_id": "test_user_123"
        }
        
        response = test_client.post("/query", json=query_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["intent"] == "multi_intent"
        assert "sub_queries" in data
        assert len(data["sub_queries"]) >= 2
    
    def test_query_endpoint_empty_message(self, test_client):
        """Test query endpoint with empty message."""
        query_data = {
            "message": ""
        }
        
        response = test_client.post("/query", json=query_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_query_endpoint_too_long_message(self, test_client):
        """Test query endpoint with too long message."""
        query_data = {
            "message": "a" * 1001  # Exceeds max length
        }
        
        response = test_client.post("/query", json=query_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_query_endpoint_invalid_json(self, test_client):
        """Test query endpoint with invalid JSON."""
        response = test_client.post(
            "/query",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_cors_headers(self, test_client):
        """Test CORS headers."""
        # Test CORS preflight request
        headers = {
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type"
        }
        response = test_client.options("/query", headers=headers)
        
        # Should allow the request
        assert response.status_code == 200
        # CORS should be configured
        assert "access-control-allow-origin" in response.headers
    
    def test_error_response_format(self, test_client):
        """Test error response format."""
        response = test_client.post("/query", json={"invalid": "data"})
        
        assert response.status_code == 422
        data = response.json()
        
        # FastAPI validation errors have different format
        assert "detail" in data
    
    def test_concurrent_queries(self, test_client):
        """Test handling multiple concurrent queries."""
        import concurrent.futures
        
        queries = [
            {"message": "Qual é o meu saldo?", "user_id": "test_user_123"},
            {"message": "O que é o InfinitePay?"},
            {"message": "Preciso de ajuda"},
        ]
        
        def send_query(query_data):
            return test_client.post("/query", json=query_data)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(send_query, query) for query in queries]
            responses = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert "answer" in data
            assert "agent_used" in data
    
    def test_response_format_consistency(self, test_client):
        """Test that all responses have consistent format."""
        test_queries = [
            {"message": "Test query 1"},
            {"message": "Test query 2", "user_id": "test_user_123"},
        ]
        
        for query_data in test_queries:
            response = test_client.post("/query", json=query_data)
            
            assert response.status_code == 200
            data = response.json()
            
            # Check required fields
            required_fields = ["answer", "agent_used", "confidence"]
            for field in required_fields:
                assert field in data
            
            # Check field types
            assert isinstance(data["answer"], str)
            assert isinstance(data["agent_used"], str)
            assert isinstance(data["confidence"], (int, float))
            assert 0 <= data["confidence"] <= 1
            
            # Optional fields should be properly typed if present
            if "sources" in data and data["sources"] is not None:
                assert isinstance(data["sources"], list)
            
            if "intent" in data and data["intent"] is not None:
                assert isinstance(data["intent"], str)
            
            if "handoff_to_human" in data:
                assert isinstance(data["handoff_to_human"], bool)
            
            if "requires_user_id" in data:
                assert isinstance(data["requires_user_id"], bool)