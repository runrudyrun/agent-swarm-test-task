"""Customer Support Agent with access to user data tools."""

import logging
from typing import Dict, List, Optional

from langchain.agents import Tool
from langchain.schema import BaseMessage, HumanMessage, SystemMessage

from tools.user_store import (
    TOOL_METADATA,
    get_account_details,
    get_recent_transactions,
    get_tool_suggestions,
    open_support_ticket,
)

logger = logging.getLogger(__name__)


class SupportAgent:
    """Agent for handling customer support queries with access to user data."""
    
    def __init__(self):
        self.tools = self._create_tools()
        self.system_prompt = self._create_system_prompt()
    
    def _create_tools(self) -> List[Tool]:
        """Create LangChain tools for the agent."""
        return [
            Tool(
                name="get_account_details",
                func=get_account_details,
                description="Obter detalhes da conta do usuário (requer user_id)"
            ),
            Tool(
                name="get_recent_transactions",
                func=get_recent_transactions,
                description="Obter transações recentes do usuário (requer user_id, opcional: limit)"
            ),
            Tool(
                name="open_support_ticket",
                func=open_support_ticket,
                description="Abrir novo ticket de suporte (requer user_id, subject, description)"
            )
        ]
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for the support agent."""
        return """Você é um assistente de suporte ao cliente da InfinitePay, especializado em ajudar usuários com questões relacionadas às suas contas e transações.

SUAS RESPONSABILIDADES:
1. Fornecer informações claras e precisas sobre contas e transações
2. Ajudar usuários a entenderem seus dados e resolver problemas
3. Criar tickets de suporte quando necessário
4. Sempre responder em português do Brasil (pt-BR)
5. Manter um tom profissional, mas amigável e empático

FERRAMENTAS DISPONÍVEIS:
- get_account_details: Obter detalhes da conta (saldo, status, dados cadastrais)
- get_recent_transactions: Obter histórico de transações recentes
- open_support_ticket: Criar novo ticket de suporte

DIRETRIZES IMPORTANTES:
- SEMPRE verifique se o user_id está disponível antes de usar ferramentas
- Se não houver user_id, peça educadamente ao usuário
- Para questões técnicas ou problemas complexos, crie um ticket
- Seja proativo em oferecer ajuda adicional
- Use emojis apropriados para tornar a comunicação mais amigável

EXEMPLOS DE RESPOSTAS:
✅ "Claro! Posso ajudar você com o extrato da sua conta. Qual é o seu ID de usuário?"
✅ "Verificando suas transações recentes..."
❌ "Não posso ajudar" (sem explicação)"""
    
    def process_query(self, query: str, user_id: Optional[str] = None) -> Dict:
        """Process a support query and return response."""
        logger.info(f"SupportAgent processing query: {query}")
        
        # Check if user_id is required but not provided
        if not user_id and self._requires_user_id(query):
            return {
                "answer": "Olá! 👋 Para ajudar você com informações da sua conta, preciso do seu ID de usuário. Você pode encontrá-lo em seu perfil ou e-mails da InfinitePay. Qual é o seu ID?",
                "agent_used": "support",
                "tool_used": None,
                "requires_user_id": True
            }
        
        # Get tool suggestions based on query
        tool_suggestions = get_tool_suggestions(query)
        logger.info(f"Tool suggestions for query '{query}': {tool_suggestions}")
        
        # Handle different types of queries
        if any(word in query.lower() for word in ["saldo", "conta", "account", "balance", "login", "sign in", "access"]):
            if user_id:
                account_info = get_account_details(user_id)
                return {
                    "answer": account_info,
                    "agent_used": "support",
                    "tool_used": "get_account_details",
                    "requires_user_id": False
                }
        
        elif any(word in query.lower() for word in ["transações", "extrato", "histórico", "movimentações"]):
            if user_id:
                # Extract limit if mentioned
                limit = self._extract_limit(query)
                transactions = get_recent_transactions(user_id, limit)
                return {
                    "answer": transactions,
                    "agent_used": "support",
                    "tool_used": "get_recent_transactions",
                    "requires_user_id": False
                }
        
        elif any(word in query.lower() for word in ["suporte", "ajuda", "problema", "ticket", "assistência"]):
            # Try to extract subject and description from query
            subject, description = self._extract_ticket_info(query)
            
            if user_id and subject and description:
                ticket_result = open_support_ticket(user_id, subject, description)
                return {
                    "answer": ticket_result,
                    "agent_used": "support",
                    "tool_used": "open_support_ticket",
                    "requires_user_id": False
                }
            else:
                return {
                    "answer": "Posso ajudar você a criar um ticket de suporte! 🎫\n\nPara registrar seu problema, preciso de:\n1. Um breve assunto (ex: 'Problema com maquininha')\n2. Descrição detalhada do problema\n\nPor favor, me diga qual é o problema que você está enfrentando.",
                    "agent_used": "support",
                    "tool_used": None,
                    "requires_user_id": user_id is None
                }
        
        # Default response for unclear queries - enhanced with LLM
        return self._handle_general_support_query(query, user_id)
    
    def _handle_general_support_query(self, query: str, user_id: Optional[str]) -> Dict:
        """Handle general support queries with intelligent responses."""
        try:
            # Use LLM to provide a more intelligent response for support queries
            from langchain.schema import HumanMessage, SystemMessage
            
            llm = self._get_llm()
            
            # Use the existing system prompt method
            system_msg = self.get_system_message()
            
            # Create a specific human message for this query
            human_prompt = f"User query: {query}"
            messages = [system_msg, HumanMessage(content=human_prompt)]
            
            response = llm.invoke(messages)
            answer = response.content.strip()
            
            return {
                "answer": answer,
                "agent_used": "support",
                "tool_used": None,
                "requires_user_id": False
            }
            
        except Exception as e:
            logger.warning(f"LLM general support failed: {e}")
            # Fallback to generic response
            return {
                "answer": "Entendo que você está enfrentando dificuldades. Posso ajudar você com:\n\n💰 **Dados da conta** - saldo, informações cadastrais\n📊 **Transações** - histórico de pagamentos e saques\n🎫 **Suporte** - criar tickets para problemas\n\nQual seria a melhor forma de ajudar você hoje? Se precisar de informações específicas da conta, me diga seu ID de usuário.",
                "agent_used": "support",
                "tool_used": None,
                "requires_user_id": False
            }
    
    def _requires_user_id(self, query: str) -> bool:
        """Check if query requires user ID."""
        support_keywords = ["saldo", "conta", "transações", "extrato", "histórico", "movimentações"]
        return any(keyword in query.lower() for keyword in support_keywords)
    
    def _extract_limit(self, query: str) -> int:
        """Extract limit number from query."""
        import re
        
        # Look for numbers in the query
        numbers = re.findall(r'\d+', query)
        if numbers:
            limit = int(numbers[0])
            # Ensure reasonable limit
            return min(max(limit, 1), 50)
        
        return 5  # Default limit
    
    def _extract_ticket_info(self, query: str) -> tuple:
        """Extract subject and description for ticket."""
        # Simple extraction - in production, use NLP
        if len(query.split()) > 5:
            # Use first sentence as subject, rest as description
            sentences = query.split('.')
            subject = sentences[0][:100]  # Limit subject length
            description = query[:1000]  # Limit description length
            return subject, description
        
        return None, None
    
    def get_tools(self) -> List[Tool]:
        """Get available tools."""
        return self.tools
    
    def get_system_message(self) -> SystemMessage:
        """Get system message for LLM integration."""
        return SystemMessage(content=self.system_prompt)
