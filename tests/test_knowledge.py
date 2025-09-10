"""Tests for Knowledge Agent."""

import pytest
from unittest.mock import Mock, patch

from agents.knowledge_agent import KnowledgeAgent


class TestKnowledgeAgent:
    """Test cases for Knowledge Agent."""
    
    def test_knowledge_agent_initialization(self):
        """Test knowledge agent initialization."""
        with patch('agents.knowledge_agent.Chroma') as mock_chroma:
            mock_chroma.return_value._collection.count.return_value = 0
            
            agent = KnowledgeAgent()
            
            assert agent is not None
            assert agent.vectorstore is None  # Should be None when no documents
            assert not agent.is_available()
    
    def test_knowledge_agent_with_mock_vectorstore(self):
        """Test knowledge agent with mock vector store."""
        with patch('agents.knowledge_agent.Chroma') as mock_chroma:
            # Mock successful vector store
            mock_vectorstore = Mock()
            mock_vectorstore._collection.count.return_value = 10
            mock_chroma.return_value = mock_vectorstore
            
            agent = KnowledgeAgent()
            
            assert agent.vectorstore is not None
            assert agent.is_available()
    
    def test_process_query_no_vectorstore(self):
        """Test processing query when vector store is not available."""
        with patch('agents.knowledge_agent.Chroma') as mock_chroma:
            mock_chroma.return_value._collection.count.return_value = 0
            
            agent = KnowledgeAgent()
            result = agent.process_query("O que é o InfinitePay?")
            
            assert "não tenho acesso" in result["answer"].lower()
            assert result["agent_used"] == "knowledge"
            assert result["confidence"] == 0.0
            assert result["sources"] == []
    
    def test_format_sources(self):
        """Test source formatting."""
        agent = KnowledgeAgent()
        
        # Mock source documents
        mock_docs = [
            Mock(metadata={"source": "https://example.com", "title": "Example Page"}),
            Mock(metadata={"source": "https://test.com", "title": ""}),
            Mock(metadata={"source": "https://example.com", "title": "Example Page"}),  # Duplicate
        ]
        
        sources = agent._format_sources(mock_docs)
        
        assert len(sources) == 2  # Should deduplicate
        # Check that the formatted sources contain the expected content
        source_contents = ''.join(sources)
        assert "Example Page" in source_contents
        assert "https://example.com" in source_contents
        assert "https://test.com" in source_contents
    
    def test_calculate_confidence(self):
        """Test confidence calculation."""
        agent = KnowledgeAgent()
        
        # Test with no documents
        confidence = agent._calculate_confidence([])
        assert confidence == 0.0
        
        # Test with mock documents
        mock_docs = [
            Mock(page_content="a" * 500),
            Mock(page_content="b" * 800),
            Mock(page_content="c" * 300),
        ]
        
        confidence = agent._calculate_confidence(mock_docs)
        assert 0.3 < confidence <= 1.0  # Should have reasonable confidence
    
    def test_system_message(self):
        """Test system message generation."""
        agent = KnowledgeAgent()
        
        system_msg = agent.get_system_message()
        
        assert system_msg is not None
        assert "InfinitePay" in system_msg.content
        assert "contexto" in system_msg.content.lower()
        assert "português" in system_msg.content.lower()
    
    def test_create_qa_prompt(self):
        """Test QA prompt creation."""
        agent = KnowledgeAgent()
        
        prompt = agent._create_qa_prompt()
        
        assert prompt is not None
        assert "CONTEXTO:" in prompt.template
        assert "PERGUNTA:" in prompt.template
        assert "RESPOSTA:" in prompt.template
        assert "português" in prompt.template.lower()
    
    @patch('agents.knowledge_agent.RetrievalQA')
    def test_process_query_with_mock_chain(self, mock_qa_class):
        """Test processing query with mocked QA chain."""
        # Mock QA chain result
        mock_result = {
            "result": "O InfinitePay é uma empresa de pagamentos que oferece maquininhas e soluções financeiras.",
            "source_documents": [
                Mock(metadata={"source": "https://example.com", "title": "Sobre"})
            ]
        }
        
        # Mock the chain and its methods
        mock_chain = Mock()
        mock_chain.invoke.return_value = mock_result
        mock_qa_class.from_chain_type.return_value = mock_chain
        
        with patch('agents.knowledge_agent.Chroma') as mock_chroma:
            mock_chroma.return_value._collection.count.return_value = 10
            
            agent = KnowledgeAgent()
            result = agent.process_query("O que é o InfinitePay?")
            
            assert "InfinitePay" in result["answer"]
            assert result["agent_used"] == "knowledge"
            assert result["confidence"] > 0
            assert len(result["sources"]) > 0
    
    def test_get_llm_openai(self):
        """Test LLM selection for OpenAI."""
        with patch.dict('os.environ', {'MODEL_PROVIDER': 'openai', 'OPENAI_API_KEY': 'test-key'}):
            with patch('langchain_openai.ChatOpenAI') as mock_openai:
                agent = KnowledgeAgent()
                llm = agent._get_llm()
                
                mock_openai.assert_called_once()
    
    def test_get_llm_local(self):
        """Test LLM selection for local/default."""
        with patch.dict('os.environ', {'MODEL_PROVIDER': 'local'}):
            agent = KnowledgeAgent()
            llm = agent._get_llm()
            
            # Should return a mock LLM
            assert llm is not None
            assert hasattr(llm, '_llm_type')
    
    def test_error_handling(self):
        """Test error handling in query processing."""
        with patch('agents.knowledge_agent.Chroma') as mock_chroma:
            mock_chroma.return_value._collection.count.return_value = 10
            
            agent = KnowledgeAgent()
            
            # Mock an error in the QA chain
            with patch.object(agent, '_get_llm', side_effect=Exception("Test error")):
                result = agent.process_query("Test query")
                
                assert "erro" in result["answer"].lower()
                assert result["confidence"] == 0.0
                assert result["sources"] == []