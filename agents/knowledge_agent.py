"""Knowledge Agent with RAG capabilities for InfinitePay content."""

import logging
from typing import Dict, List, Optional

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain_chroma import Chroma

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
    
    def _create_qa_prompt(self, lang: str = "pt") -> PromptTemplate:
        """Create QA prompt template."""
        language_map = {
            "en": "English",
            "pt": "Portuguese"
        }
        output_language = language_map.get(lang.split('-')[0], "Portuguese")

        template = f"""You are a knowledgeable and friendly assistant for InfinitePay. Your primary goal is to provide clear, helpful, and accurate answers based on the CONTEXT provided.

CONTEXT:
{{context}}

QUESTION:
{{question}}

INSTRUCTIONS:
1.  **Language**: You MUST respond in the following language: **{output_language}**.
2.  **Tone**: Be natural, conversational, and friendly. Use markdown and emojis to improve readability.
3.  **Scope**: If the user's question is unrelated to InfinitePay (e.g., sports, news, personal questions), you MUST politely state that you can only answer questions about InfinitePay products and services. Do NOT answer the off-topic question.
4.  **Context Usage**: Base your answer strictly on the CONTEXT provided. Do not use prior knowledge.
5.  **No Information**: If the CONTEXT does not contain the answer, state that you don't have that specific information and suggest contacting InfinitePay support or visiting the official website.
6.  **Citations**: When you use information from the context, cite the sources provided.

ANSWER:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def process_query(self, query: str, lang: str = "pt") -> Dict:
        """Process a knowledge query and return response."""
        logger.info(f"KnowledgeAgent processing query: {query} (lang: {lang})")
        
        try:
            # Check if vector store has content
            if not self.vectorstore or not self._has_sufficient_content():
                return self._handle_no_content(query, lang=lang)
            
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
                    return self._handle_no_relevant_content(query, lang=lang)
            except Exception as e:
                logger.warning(f"Retrieval failed: {e}")
                return self._handle_no_relevant_content(query, lang=lang)
            
            # Create QA chain with retrieved documents
            qa_chain = RetrievalQA.from_chain_type(
                llm=self._get_llm(),
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={
                    "prompt": self._create_qa_prompt(lang=lang)
                }
            )
            
            # Run query
            result = qa_chain.invoke({"query": query})
            
            # Extract answer and sources
            answer = result.get("result", "Desculpe, não consegui processar sua pergunta.")
            source_docs = result.get("source_documents", [])
            
            # Check if we got meaningful content
            if not source_docs or self._is_answer_insufficient(answer):
                return self._handle_no_relevant_content(query, lang=lang)
            
            # Format sources
            sources = self._format_sources(source_docs)
            
            # Calculate confidence based on source relevance
            confidence = self._calculate_confidence(source_docs)
            
            # Add sources to the answer only if the answer is not a refusal
            if sources and not self._is_answer_insufficient(answer):
                answer += f"\n\n📚 **Fontes:**\n" + "\n".join(sources)
            
            return {
                "answer": answer,
                "agent_used": "knowledge",
                "sources": sources,
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Error processing knowledge query: {e}")
            return self._handle_fallback_response(query, lang=lang)
    
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
            # General lack of information
            "não tenho acesso", "não consegui processar", "não tenho informações",
            "não há informações suficientes", "i don't have access", "i don't have information",
            "no context provided", "no information available",
            # Off-topic refusals (more general)
            "só posso responder", "i can only answer",
            "não posso responder", "i cannot answer"
        ]
        return any(pattern in answer.lower() for pattern in insufficient_patterns)
    
    def _handle_no_content(self, query: str, lang: str = "pt") -> Dict:
        """Handle case when no vector store content is available."""
        language_map = {
            "en": "English",
            "pt": "Portuguese"
        }
        output_language = language_map.get(lang.split('-')[0], "Portuguese")

        # This is a system-level failure, so we use a predefined response instead of the LLM.
        if output_language == "English":
            answer = "I'm sorry, my knowledge base is currently unavailable, so I can't answer your question at the moment. Please try again in a few moments. Thank you for your patience! 🙏"
        else:
            answer = "Desculpe, minha base de conhecimento está indisponível no momento, então não consigo responder sua pergunta agora. Por favor, tente novamente em alguns instantes. Agradeço a sua paciência! 🙏"

        return {
            "answer": answer,
            "agent_used": "knowledge",
            "sources": [],
            "confidence": 0.05,
            "note": "Vector store content unavailable."
        }
    
    def _handle_no_relevant_content(self, query: str, lang: str = "pt") -> Dict:
        """Handle case when no relevant RAG documents are found by using the LLM to politely decline."""
        try:
            llm = self._get_llm()

            language_map = {
                "en": "English",
                "pt": "Portuguese"
            }
            output_language = language_map.get(lang.split('-')[0], "Portuguese")

            prompt = (
                f"You are an assistant for InfinitePay. The user asked a question that is likely off-topic: '{query}'. "
                "Your knowledge base had no relevant information. "
                f"Your task is to politely inform the user that you can only answer questions about InfinitePay products and services. "
                f"Do NOT attempt to answer the user's question. You MUST respond in **{output_language}**."
            )

            response = llm.invoke(prompt)
            answer = response.content.strip()

            return {
                "answer": answer,
                "agent_used": "knowledge",
                "sources": [],
                "confidence": 0.1,
                "note": "No relevant RAG content found; LLM generated off-topic response."
            }
        except Exception as e:
            logger.error(f"Fallback LLM response for no relevant content failed: {e}")
            return self._handle_fallback_response(query, lang=lang)
    
    def _handle_fallback_response(self, query: str, lang: str = "pt") -> Dict:
        """Final fallback response when all else fails."""
        if lang.startswith("en"):
            answer = "I'm sorry, I couldn't find specific information about your question. I recommend contacting InfinitePay support for personalized assistance."
        else:
            answer = "Desculpe, não consegui encontrar informações específicas sobre sua pergunta. Recomendo entrar em contato com o suporte da InfinitePay para obter ajuda personalizada."

        return {
            "answer": answer,
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