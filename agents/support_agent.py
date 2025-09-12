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
    UserStore,
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
    
    def _create_system_prompt(self, lang: str = "pt") -> str:
        """Create system prompt for the support agent."""
        language_map = {
            "en": "English",
            "pt": "Portuguese"
        }
        output_language = language_map.get(lang.split('-')[0], "Portuguese")

        return f"""You are a customer support assistant for InfinitePay, specializing in helping users with account and transaction-related issues.

YOUR RESPONSIBILITIES:
1.  **Language**: You MUST respond in the following language: **{output_language}**.
2.  Provide clear and accurate information about accounts and transactions.
3.  Help users understand their data and resolve issues.
4.  Create support tickets when necessary.
5.  Maintain a professional, yet friendly and empathetic tone.
AN
AVAILABLE TOOLS:
- get_account_details: Get user account details (balance, status, registration info).
- get_recent_transactions: Get recent transaction history.
- open_support_ticket: Create a new support ticket.

IMPORTANT GUIDELINES:
- ALWAYS check if a user_id is available before using tools.
- If user_id is missing, politely explain that there is a technical issue, respond in ({output_language}).
- For technical or complex issues, create a ticket.
- Be proactive in offering additional help.
- Use appropriate emojis to make the communication friendlier.
- Do not just provide technical data about the account data and transactions, use them to address the client question.
"""
    
    def process_query(self, query: str, user_id: Optional[str] = None, lang: str = "pt") -> Dict:
        """Process a support query and return response."""
        logger.info(f"SupportAgent processing query: {query}")

        
        # Get tool suggestions based on query
        tool_suggestions = get_tool_suggestions(query)
        logger.info(f"Tool suggestions for query '{query}': {tool_suggestions}")
        
        # If the query requires user context but user_id is missing, ask for it explicitly
        if self._requires_user_id(query) and not user_id:
            ask_map = {
                "en": (
                    "To check your account or transactions I need your user ID. "
                    "Please provide your user_id so I can securely verify your information and help you faster."
                ),
                "pt": (
                    "Para verificar sua conta ou transações, preciso do seu ID de usuário. "
                    "Por favor, me informe seu user_id para que eu possa verificar com segurança e ajudar mais rápido."
                ),
            }
            answer = ask_map.get(lang.split('-')[0], ask_map["pt"])
            return {
                "answer": answer,
                "agent_used": "support",
                "tool_used": None,
                "requires_user_id": True,
            }
        
        # Handle different types of queries
        if any(word in query.lower() for word in [
            "saldo", "conta", "account", "balance", "login", "sign in", "access",
            "perfil", "profile"
        ]):
            if user_id:
                account_info = get_account_details(user_id)
                return {
                    "answer": account_info,
                    "agent_used": "support",
                    "tool_used": "get_account_details",
                    "requires_user_id": False
                }
        
        elif any(word in query.lower() for word in [
            "transações", "extrato", "histórico", "movimentações",
            "transactions", "statement", "history"
        ]):
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

        # Transfer-related diagnostics (English and Portuguese)
        elif any(word in query.lower() for word in [
            "transfer", "transfers", "transferência", "transferências", "transferir", "pix"
        ]):
            if user_id:
                try:
                    store = UserStore()
                    user = store.get_user_by_id(user_id)
                    if not user:
                        not_found_map = {
                            "en": f"User {user_id} not found.",
                            "pt": f"❌ Usuário {user_id} não encontrado.",
                        }
                        answer = not_found_map.get(lang.split('-')[0], not_found_map["pt"])
                        return {
                            "answer": answer,
                            "agent_used": "support",
                            "tool_used": None,
                            "requires_user_id": False,
                        }

                    status = user.get("status", "unknown")
                    balance = user.get("balance", 0)
                    # Get recent transactions for signal (failures/pending)
                    recent = store.get_user_transactions(user_id, limit=5)
                    failed_count = sum(1 for t in recent if t.get("status") == "failed")
                    pending_count = sum(1 for t in recent if t.get("status") == "pending")

                    # Prepare facts for LLM summarization
                    balance_str = f"R$ {balance:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                    facts = {
                        "account_status": status,
                        "available_balance": balance_str,
                        "recent_pending_transactions": pending_count,
                        "recent_failed_transactions": failed_count,
                        "channel": "transfers/PIX",
                        "recommended_actions": [
                            "Confirmar dados/identidade se solicitado pelo app",
                            "Resolver eventuais pendências de conformidade/segurança",
                            "Conferir dados do destinatário e valor",
                            "Verificar se há atualização pendente do app",
                            "Tentar novamente em alguns minutos"
                        ],
                    }

                    # Try to summarize with LLM using the verified facts
                    summarized = self._summarize_support_facts_with_llm(query, facts, lang=lang)
                    if summarized:
                        return {
                            "answer": summarized,
                            "agent_used": "support",
                            "tool_used": "diagnose_transfers",
                            "requires_user_id": False,
                        }

                    # Language-specific templates (fallback if LLM unavailable)
                    if lang.startswith("en"):
                        if status != "active":
                            answer = (
                                "I checked your account and transfers are currently unavailable because your account "
                                f"status is '{status}'. This usually blocks PIX and bank transfers.\n\n"
                                "What you can do now:\n"
                                "- Confirm your registration/identity (if requested in the app).\n"
                                "- Resolve any pending compliance or security review.\n"
                                "- If you need, I can open a support ticket right away to speed this up."
                            )
                        else:
                            answer = (
                                "Your account is active. Here is a quick check to understand transfers: \n"
                                f"- Available balance: {balance_str}.\n"
                                f"- Recent transactions: {pending_count} pending, {failed_count} failed.\n\n"
                                "If you're seeing errors when transferring, please try: \n"
                                "1) Confirm the recipient data and amount.\n"
                                "2) Check if there are any app updates pending.\n"
                                "3) Try again in a few minutes (temporary network issues).\n\n"
                                "Want me to open a support ticket describing this transfer issue for you?"
                            )
                    else:
                        if status != "active":
                            answer = (
                                "Verifiquei sua conta e as transferências estão indisponíveis porque o status da sua conta "
                                f"é '{status}'. Isso normalmente bloqueia PIX e transferências bancárias.\n\n"
                                "O que você pode fazer agora:\n"
                                "- Confirmar seus dados/identidade (se o app solicitar).\n"
                                "- Resolver pendências de conformidade ou revisão de segurança.\n"
                                "- Se quiser, eu já abro um ticket de suporte para agilizar."
                            )
                        else:
                            answer = (
                                "Sua conta está ativa. Aqui vai um check rápido para entender as transferências: \n"
                                f"- Saldo disponível: {balance_str}.\n"
                                f"- Transações recentes: {pending_count} pendentes, {failed_count} com falha.\n\n"
                                "Se você estiver vendo erro ao transferir, tente: \n"
                                "1) Confirmar os dados do destinatário e o valor.\n"
                                "2) Verificar se há atualização pendente do app.\n"
                                "3) Tentar novamente em alguns minutos (instabilidade temporária).\n\n"
                                "Quer que eu abra um ticket de suporte descrevendo esse problema de transferência para você?"
                            )

                    return {
                        "answer": answer,
                        "agent_used": "support",
                        "tool_used": "diagnose_transfers",
                        "requires_user_id": False,
                    }
                except Exception as e:
                    logger.warning(f"Transfer diagnostics failed: {e}")
                    # Fallback to general handler
                    return self._handle_general_support_query(query, user_id, lang=lang)
        
        elif any(word in query.lower() for word in ["suporte", "ajuda", "problema", "ticket", "assistência", "help", "problem"]):
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
                answer_map = {
                    "en": "I can help you create a support ticket! 🎫\n\nTo log your issue, I need:\n1. A brief subject (e.g., 'Problem with card machine')\n2. A detailed description of the problem\n\nPlease tell me what issue you are facing.",
                    "pt": "Posso ajudar você a criar um ticket de suporte! 🎫\n\nPara registrar seu problema, preciso de:\n1. Um breve assunto (ex: 'Problema com maquininha')\n2. Descrição detalhada do problema\n\nPor favor, me diga qual é o problema que você está enfrentando."
                }
                answer = answer_map.get(lang.split('-')[0], answer_map["pt"])
                return {
                    "answer": answer,
                    "agent_used": "support",
                    "tool_used": None,
                    "requires_user_id": user_id is None
                }
        
        # Default response for unclear queries - enhanced with LLM
        return self._handle_general_support_query(query, user_id, lang=lang)
    
    def _get_llm(self):
        """Get LLM instance for intelligent responses."""
        from rag.config import create_llm
        return create_llm()
    
    def _handle_general_support_query(self, query: str, user_id: Optional[str], lang: str = "pt") -> Dict:
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
            answer_map = {
                "en": "I understand you're facing difficulties. I can help you with:\n\n💰 **Account data** - balance, registration information\n📊 **Transactions** - history of payments and withdrawals\n🎫 **Support** - create tickets for issues\n\nHow can I best help you today? If you need specific account information, please provide your user ID.",
                "pt": "Entendo que você está enfrentando dificuldades. Posso ajudar você com:\n\n💰 **Dados da conta** - saldo, informações cadastrais\n📊 **Transações** - histórico de pagamentos e saques\n🎫 **Suporte** - criar tickets para problemas\n\nQual seria a melhor forma de ajudar você hoje? Se precisar de informações específicas da conta, me diga seu ID de usuário."
            }
            answer = answer_map.get(lang.split('-')[0], answer_map["pt"])
            return {
                "answer": answer,
                "agent_used": "support",
                "tool_used": None,
                "requires_user_id": False
            }
    
    def _requires_user_id(self, query: str) -> bool:
        """Check if query requires user ID."""
        support_keywords = [
            # Portuguese
            "saldo", "conta", "transações", "extrato", "histórico", "movimentações", "transferência", "transferências",
            # English
            "account", "balance", "transactions", "statement", "transfer", "transfers", "login", "sign in"
        ]
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
    
    def get_system_message(self, lang: str = "pt") -> SystemMessage:
        """Get system message for LLM integration."""
        return SystemMessage(content=self._create_system_prompt(lang=lang))

    def _summarize_support_facts_with_llm(self, query: str, facts: Dict, lang: str = "pt") -> Optional[str]:
        """Use the LLM to paraphrase verified support facts without altering them.
        Returns None if LLM is unavailable or fails.
        """
        try:
            llm = self._get_llm()
            import json

            language_map = {
                "en": "English",
                "pt": "Portuguese",
            }
            output_language = language_map.get(lang.split('-')[0], "Portuguese")

            system_prompt = (
                f"You are a customer support assistant for InfinitePay. You will receive a user query and a set of VERIFIED FACTS "
                f"about the user's account or transactions. Your job is to write a clear, friendly, and helpful response in {output_language} that:\n"
                "- Uses ONLY the provided facts.\n"
                "- DOES NOT invent numbers, statuses, or actions.\n"
                "- Does not reveal raw JSON; summarize succinctly.\n"
                "- Offers practical next steps and the option to open a support ticket.\n"
                "- Keeps the response concise and readable, using bullet points where helpful.\n"
            )

            human_prompt = (
                "USER QUERY:\n" + query + "\n\n" +
                "FACTS (do not alter):\n" + json.dumps(facts, ensure_ascii=False, indent=2)
            )

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt),
            ]

            response = llm.invoke(messages)
            text = (response.content or "").strip()
            if not text:
                return None
            return text
        except Exception as e:
            logger.warning(f"LLM fact summarization failed: {e}")
            return None
