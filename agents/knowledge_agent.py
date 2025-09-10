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
        
        if not self.vectorstore:
            return {
                "answer": "Desculpe, não tenho acesso ao banco de dados de conhecimento no momento. Por favor, tente novamente mais tarde ou entre em contato com o suporte.",
                "agent_used": "knowledge",
                "sources": [],
                "confidence": 0.0
            }
        
        try:
            # Create retriever with MMR for diversity
            retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": RAGConfig.MMR_K,
                    "fetch_k": RAGConfig.MMR_FETCH_K
                }
            )
            
            # Create QA chain
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
            return {
                "answer": "Desculpe, ocorreu um erro ao processar sua pergunta. Por favor, tente novamente ou entre em contato com o suporte.",
                "agent_used": "knowledge",
                "sources": [],
                "confidence": 0.0
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
        import os
        
        provider = os.getenv("MODEL_PROVIDER", "local")
        
        if provider == "openai" and os.getenv("OPENAI_API_KEY"):
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(temperature=0.1, model="gpt-3.5-turbo")
        else:
            # Fallback to a simple local model or mock
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
                
                @property
                def _llm_type(self):
                    return "mock"
            
            return MockLLM()
    
    def is_available(self) -> bool:
        """Check if knowledge agent is available (vector store loaded)."""
        return self.vectorstore is not None
    
    def get_system_message(self) -> SystemMessage:
        """Get system message for LLM integration."""
        return SystemMessage(content=self.system_prompt)