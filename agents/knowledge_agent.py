"""Knowledge Agent with RAG capabilities for InfinitePay content."""

import logging
from typing import Dict, List, Optional

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain_community.vectorstores import Chroma

from rag.config import RAGConfig, get_embeddings

logger = logging.getLogger(__name__)


class KnowledgeAgent:
    """Agent for answering questions using RAG over InfinitePay content."""
    
    def __init__(self):
        self.vectorstore = None
        self.qa_chain = None
        self.system_prompt = self._create_system_prompt()
        self._initialize_vectorstore()
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for the knowledge agent."""
        return """Você é um assistente de conhecimento da InfinitePay, especializado em responder perguntas sobre produtos, serviços, taxas, funcionalidades e informações gerais da empresa.

SUAS RESPONSABILIDADES:
1. Responder perguntas usando APENAS as informações fornecidas no contexto
2. Se a informação não estiver no contexto, diga claramente que não tem essa informação
3. Fornecer respostas claras, concisas e úteis em português do Brasil
4. Sempre citar as fontes utilizadas quando possível
5. Manter um tom profissional e amigável

DIRETRIZES IMPORTANTES:
- Use SOMENTE informações do contexto fornecido
- Se não houver informação suficiente, ofereça pesquisar na web (se disponível)
- Inclua URLs das fontes quando relevante
- Seja específico sobre produtos e serviços
- Use exemplos práticos quando apropriado

FORMATO DAS RESPOSTAS:
- Use markdown para melhor legibilidade
- Inclua seções claramente marcadas
- Adicione emojis apropriados para tornar mais amigável
- Liste fontes no final quando aplicável"""
    
    def _initialize_vectorstore(self):
        """Initialize the vector store and QA chain."""
        try:
            # Try to load existing vector store
            self.vectorstore = Chroma(
                persist_directory=RAGConfig.VECTOR_STORE_PATH,
                embedding_function=get_embeddings(),
                collection_name=RAGConfig.COLLECTION_NAME
            )
            
            # Test if collection exists and has documents
            try:
                count = self.vectorstore._collection.count()
                if count == 0:
                    logger.warning("Vector store exists but contains no documents")
                    self.vectorstore = None
                else:
                    logger.info(f"Vector store loaded with {count} documents")
            except Exception as e:
                logger.warning(f"Vector store test failed: {e}")
                self.vectorstore = None
                
        except Exception as e:
            logger.warning(f"Failed to load vector store: {e}")
            self.vectorstore = None
    
    def _create_qa_prompt(self) -> PromptTemplate:
        """Create QA prompt template."""
        template = """Você é um assistente de conhecimento da InfinitePay. Use as seguintes informações do contexto para responder à pergunta do usuário.

CONTEXTO:
{context}

PERGUNTA: {question}

INSTRUÇÕES:
1. Responda APENAS com base nas informações do contexto acima
2. Se a informação não estiver disponível no contexto, diga claramente que não tem essa informação
3. Forneça respostas detalhadas mas concisas
4. Use formato markdown para melhor legibilidade
5. Inclua URLs das fontes quando mencionar produtos ou serviços
6. Responda sempre em português do Brasil
7. Use emojis apropriados para tornar a resposta mais amigável

RESPOSTA:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def process_query(self, query: str) -> Dict:
        """Process a knowledge query and return response."""
        logger.info(f"KnowledgeAgent processing query: {query}")
        
        try:
            # Check if vector store has content
            if not self.vectorstore or not self._has_sufficient_content():
                return self._handle_no_content(query)
            
            # Create retriever with MMR for diversity
            retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": RAGConfig.MMR_K,
                    "fetch_k": RAGConfig.MMR_FETCH_K
                }
            )
            
            # Try to retrieve relevant documents
            try:
                docs = retriever.invoke(query)
                if not docs or len(docs) == 0:
                    return self._handle_no_relevant_content(query)
            except Exception as e:
                logger.warning(f"Retrieval failed: {e}")
                return self._handle_no_relevant_content(query)
            
            # Create QA chain with retrieved documents
            qa_chain = RetrievalQA.from_chain_type(
                llm=self._get_llm(),
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={
                    "prompt": self._create_qa_prompt()
                }
            )
            
            # Run query
            result = qa_chain.invoke({"query": query})
            
            # Extract answer and sources
            answer = result.get("result", "Desculpe, não consegui processar sua pergunta.")
            source_docs = result.get("source_documents", [])
            
            # Check if we got meaningful content
            if not source_docs or self._is_answer_insufficient(answer):
                return self._handle_no_relevant_content(query)
            
            # Format sources
            sources = self._format_sources(source_docs)
            
            # Calculate confidence based on source relevance
            confidence = self._calculate_confidence(source_docs)
            
            # Add sources to answer if available
            if sources:
                answer += f"\n\n📚 **Fontes:**\n" + "\n".join(sources)
            
            return {
                "answer": answer,
                "agent_used": "knowledge",
                "sources": sources,
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Error processing knowledge query: {e}")
            return self._handle_fallback_response(query)
    
    def _has_sufficient_content(self) -> bool:
        """Check if vector store has sufficient content."""
        try:
            if not self.vectorstore:
                return False
            
            # Try to get a sample count
            try:
                count = self.vectorstore._collection.count()
                return count > 10  # Require at least 10 documents
            except Exception:
                return False
        except Exception:
            return False
    
    def _is_answer_insufficient(self, answer: str) -> bool:
        """Check if the answer is insufficient or indicates missing context."""
        insufficient_patterns = [
            "não tenho acesso",
            "não consegui processar",
            "não tenho informações",
            "não há informações suficientes",
            "i don't have access",
            "i don't have information",
            "no context provided",
            "no information available"
        ]
        return any(pattern in answer.lower() for pattern in insufficient_patterns)
    
    def _handle_no_content(self, query: str) -> Dict:
        """Handle case when no vector store content is available."""
        try:
            # Use LLM directly without RAG context
            llm = self._get_llm()
            
            # Create a prompt that asks the LLM to answer based on general knowledge
            # while being transparent about the limitation
            prompt = f'''You are a helpful assistant for InfinitePay. The user asked: {query}

Please provide a helpful response based on your general knowledge about payment systems and financial services. If you're not sure about specific InfinitePay details, be transparent about this and suggest the user contact InfinitePay support for accurate information.

Guidelines:
- Be helpful but honest about information sources
- Suggest contacting support for specific details
- Provide general guidance when possible
- Always be transparent about limitations'''
            
            response = llm.invoke(prompt)
            answer = response.content.strip()
            
            return {
                "answer": answer,
                "agent_used": "knowledge",
                "sources": ["LLM general knowledge (vector store unavailable)"],
                "confidence": 0.6,
                "note": "Answer based on general knowledge - vector store content unavailable"
            }
            
        except Exception as e:
            logger.error(f"Fallback LLM response failed: {e}")
            return self._handle_fallback_response(query)
    
    def _handle_no_relevant_content(self, query: str) -> Dict:
        """Handle case when no relevant documents are found."""
        return self._handle_no_content(query)  # Use same fallback logic
    
    def _handle_fallback_response(self, query: str) -> Dict:
        """Final fallback response when all else fails."""
        return {
            "answer": "Desculpe, não consegui encontrar informações específicas sobre sua pergunta. Recomendo entrar em contato com o suporte da InfinitePay para obter ajuda personalizada.",
            "agent_used": "knowledge",
            "sources": [],
            "confidence": 0.2,
            "note": "Fallback response - all methods failed"
        }
    
    def _format_sources(self, source_docs) -> List[str]:
        """Format source documents for response."""
        sources = []
        seen_urls = set()
        
        for doc in source_docs:
            metadata = doc.metadata
            url = metadata.get("source", "")
            title = metadata.get("title", "")
            
            if url and url not in seen_urls:
                seen_urls.add(url)
                if title:
                    sources.append(f"- [{title}]({url})")
                else:
                    sources.append(f"- {url}")
        
        return sources
    
    def _calculate_confidence(self, source_docs) -> float:
        """Calculate confidence score based on source documents."""
        if not source_docs:
            return 0.0
        
        # Simple confidence based on number and relevance of sources
        base_confidence = min(len(source_docs) * 0.2, 1.0)
        
        # Additional confidence based on content length (proxy for detail)
        total_content_length = sum(len(doc.page_content) for doc in source_docs)
        content_confidence = min(total_content_length / 2000, 0.3)
        
        return min(base_confidence + content_confidence, 1.0)
    
    def _get_llm(self):
        """Get LLM instance based on configuration."""
        from rag.config import create_llm
        return create_llm()
    
    def is_available(self) -> bool:
        """Check if knowledge agent is available (vector store loaded)."""
        return self.vectorstore is not None
    
    def get_system_message(self) -> SystemMessage:
        """Get system message for LLM integration."""
        return SystemMessage(content=self.system_prompt)