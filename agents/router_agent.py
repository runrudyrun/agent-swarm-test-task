"""Router Agent for intent classification and orchestration."""

import logging
import re
from typing import Dict, List, Optional, Tuple


from agents.knowledge_agent import KnowledgeAgent
from agents.support_agent import SupportAgent

logger = logging.getLogger(__name__)


class RouterAgent:
    """Router agent for classifying intents and routing to appropriate agents."""
    
    def __init__(self):
        self.knowledge_agent = KnowledgeAgent()
        self.support_agent = SupportAgent()
        
        # Define intent patterns for rule-based classification
        self.intent_patterns = {
            "support": [
                # Account-related
                r"saldo\b", r"conta\b", r"extrato\b", r"transações?\b", r"movimentações?\b",
                r"histórico\b", r"pagamentos?\b", r"saques?\b", r"depósitos?\b",
                # Transfers
                r"transferências?\b", r"transfer\b", r"transfers\b", r"pix\b",
                
                # User data
                r"meus?\s+dados\b", r"informações?\s+d[ao]\s+cadastro\b", r"perfil\b",
                r"meu\s+id\b", r"minha\s+conta\b",
                
                # Support
                r"suporte\b", r"ajuda\b", r"problema\b", r"dúvida\b", r"tickets?\b",
                r"reclamação\b", r"assistência\b", r"atendimento\b", r"falar\s+com\s+atendente\b",
                
                # Issues
                r"não\s+consigo\b", r"erro\b", r"falha\b", r"problema\s+com\b", r"minha\s+maquininha\b"
            ],
            "knowledge": [
                # Products and services
                r"maquininha\b", r"tap\s+to\s+pay\b", r"pdv\b", r"point\s+of\s+sale\b",
                r"link\s+de\s+pagamento\b", r"loja\s+online\b", r"boleto\b", r"conta\s+digital\b",
                r"conta\s+pj\b", r"pix\b", r"pix\s+parcelado\b", r"empréstimo\b", r"cartão\b",
                r"rendimento\b", r"taxas?\b", r"tarifas?\b", r"preços?\b", r"valores?\b",
                
                # How it works
                r"como\s+funciona\b", r"como\s+usar\b", r"funcionalidades?\b", r"benefícios?\b",
                r"vantagens?\b", r"prazos?\b", r"limites?\b", r"requisitos?\b",
                
                # Company info
                r"infinitepay\b", r"sobre\b", r"informações?\s+sobre\b", r"o\s+que\s+é\b",
                r"como\s+começar\b", r"cadastro\b", r"abrir\s+conta\b"
            ]
        }
        
        # Escalation patterns
        self.escalation_patterns = [
            r"urgente\b", r"emergência\b", r"crítico\b", r"imediatamente\b",
            r"falar\s+com\s+humano\b", r"atendente\s+humano\b", r"pessoa\s+real\b",
            r"não\s+entendi\b", r"não\s+respondeu\b", r"tentativas?\b", r"já\s+tentei\b"
        ]
        
        # Multi-intent split patterns
        self.split_patterns = [
            r"\be\b", r"\b[eé]\s+também\b", r"\balém\s+disso\b", r"\boutra\s+coisa\b",
            r"\bporém\b", r"\bmas\b", r"\bentretanto\b"
        ]
    
    def classify_intent(self, query: str) -> Tuple[str, float]:
        """Classify intent using LLM first, fallback to rule-based approach."""
        query_lower = query.lower().strip()
        
        # Check for escalation first (rule-based for safety)
        if self._check_escalation(query_lower):
            return "escalate", 1.0
        
        # Try LLM classification first if available
        llm_result = self._classify_with_llm(query)
        if llm_result:
            return llm_result
        
        # Fallback: rule-based classification
        return self._classify_with_rules(query_lower)
    
    def _classify_with_llm(self, query: str) -> Optional[Tuple[str, float]]:
        """Use LLM for intelligent intent classification."""
        try:
            import os
            from rag.config import create_llm
            from langchain.schema import HumanMessage, SystemMessage
            
            # Check if OpenAI is properly configured
            if os.getenv("MODEL_PROVIDER") != "openai" or not os.getenv("OPENAI_API_KEY"):
                return None
            
            # Create LLM instance
            llm = create_llm()
            
            # Create classification prompt
            system_prompt = '''You are an intelligent intent classifier for a customer service system. Analyze the user query and classify it into one of these categories:

INTENT CATEGORIES:
- "support": Account issues, login problems, transaction queries, balance inquiries, transfer issues, account access problems
- "knowledge": Product information, fees, rates, how-to questions, general company information, pricing, features
- "escalate": When user explicitly asks for human agent, expresses frustration, or needs urgent complex help
- "unknown": When intent is unclear or doesn't fit other categories

CLASSIFICATION RULES:
1. Consider context and synonyms
2. Account-related issues go to "support"
3. Product/company questions go to "knowledge"
4. Technical problems go to "support"
5. Explicit human agent requests go to "escalate"

RESPOND WITH:
CLASSIFICATION: <intent>
CONFIDENCE: <0.0-1.0>
REASON: <brief explanation>'''
            
            human_prompt = f"User query: {query}"
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            response = llm.invoke(messages)
            result = response.content.strip()
            
            # Parse LLM response
            lines = result.split('\n')
            intent = "unknown"
            confidence = 0.5
            
            for line in lines:
                if "CLASSIFICATION:" in line:
                    intent = line.split("CLASSIFICATION:")[1].strip().lower()
                elif "CONFIDENCE:" in line:
                    try:
                        confidence = float(line.split("CONFIDENCE:")[1].strip())
                    except ValueError:
                        confidence = 0.5
            
            # Validate intent
            if intent in ["support", "knowledge", "escalate", "unknown"]:
                return intent, min(confidence, 0.95)  # Cap at 0.95 for LLM
            
            return None
            
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")
            return None
    
    def _classify_with_rules(self, query_lower: str) -> Tuple[str, float]:
        """Fallback rule-based classification."""
        # Count matches for each intent
        intent_scores = {}
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1
            intent_scores[intent] = score
        
        # Determine primary intent
        if not intent_scores or max(intent_scores.values()) == 0:
            # Fallback: check for specific keywords
            if self._has_support_keywords(query_lower):
                return "support", 0.6
            elif self._has_knowledge_keywords(query_lower):
                return "knowledge", 0.6
            else:
                return "unknown", 0.3
        
        # Get intent with highest score
        primary_intent = max(intent_scores, key=intent_scores.get)
        max_score = intent_scores[primary_intent]
        
        # Calculate confidence based on score and query length
        confidence = min(max_score / 2, 1.0)
        
        # Boost confidence for longer queries with matches
        if len(query_lower.split()) > 3 and max_score >= 1:
            confidence = min(confidence + 0.3, 1.0)
        
        # Ensure minimum confidence for clear intent
        if max_score >= 1:
            confidence = max(confidence, 0.6)
        
        return primary_intent, confidence
    
    def route_query(self, query: str, user_id: Optional[str] = None, lang: Optional[str] = None) -> Dict:
        """Route query to appropriate agent and return response."""
        logger.info(f"RouterAgent processing query: {query}")

        # Detect language if not provided
        if lang is None:
            try:
                from langdetect import detect, LangDetectException
                lang = detect(query)
            except LangDetectException:
                lang = "pt"  # Default to Portuguese if detection fails
            logger.info(f"Detected language: {lang}")

        # Check for multi-intent queries
        sub_queries = self._split_multi_intent(query)

        if len(sub_queries) > 1:
            return self._handle_multi_intent(sub_queries, user_id, lang)

        # Single intent - classify and route
        intent, confidence = self.classify_intent(query)
        logger.info(f"Classified intent: {intent} (confidence: {confidence})")

        # Route based on intent
        if intent == "escalate":
            return {
                "answer": "Entendo que sua solicitação pode ser urgente ou complexa. Vou encaminhar você para um atendente humano que poderá ajudar melhor. Por favor, aguarde um momento... 🔄",
                "agent_used": "router",
                "intent": intent,
                "confidence": confidence,
                "lang": lang,
                "handoff_to_human": True
            }

        elif intent == "support":
            # Route to support agent
            support_response = self.support_agent.process_query(query, user_id, lang=lang)
            support_response.update({
                "intent": intent,
                "confidence": confidence,
                "lang": lang
            })
            return support_response

        elif intent == "knowledge":
            # Route to knowledge agent
            knowledge_response = self.knowledge_agent.process_query(query, lang=lang)
            knowledge_response.update({
                "intent": intent,
                "confidence": confidence,
                "lang": lang
            })
            return knowledge_response

        else:  # unknown intent
            # Try knowledge agent first (broader scope)
            if self.knowledge_agent.is_available():
                knowledge_response = self.knowledge_agent.process_query(query, lang=lang)
                if knowledge_response.get("confidence", 0) > 0.3:
                    knowledge_response.update({
                        "intent": "knowledge",
                        "confidence": confidence * 0.8,
                        "lang": lang
                    })
                    return knowledge_response

            # Fallback to support agent
            support_response = self.support_agent.process_query(query, user_id, lang=lang)
            support_response.update({
                "intent": "support",
                "confidence": confidence * 0.8,
                "lang": lang
            })
            return support_response
    
    def _check_escalation(self, query: str) -> bool:
        """Check if query should be escalated to human."""
        for pattern in self.escalation_patterns:
            if re.search(pattern, query):
                return True
        return False
    
    def _has_support_keywords(self, query: str) -> bool:
        """Check if query has support-related keywords."""
        support_keywords = ["minha", "meu", "conta", "saldo", "transação", "problema", "ajuda"]
        return any(keyword in query for keyword in support_keywords)
    
    def _has_knowledge_keywords(self, query: str) -> bool:
        """Check if query has knowledge-related keywords."""
        knowledge_keywords = ["o que é", "como funciona", "taxa", "preço", "produto", "serviço"]
        return any(keyword in query for keyword in knowledge_keywords)
    
    def _split_multi_intent(self, query: str) -> List[str]:
        """Split multi-intent queries into sub-queries."""
        # Simple splitting based on conjunctions
        query_lower = query.lower()
        
        # Find split points
        split_points = []
        for pattern in self.split_patterns:
            for match in re.finditer(pattern, query_lower):
                split_points.append(match.start())
        
        if not split_points:
            return [query]
        
        # Split query
        sub_queries = []
        start = 0
        for point in sorted(split_points):
            if point - start > 5:  # Reduced minimum length for sub-query
                sub_query = query[start:point].strip()
                if sub_query:
                    sub_queries.append(sub_query)
            
            # Move past the conjunction word
            remaining_text = query_lower[point:]
            words_after = remaining_text.split()
            if words_after:
                first_word = words_after[0]
                start = point + len(first_word)
            else:
                start = point + 1
        
        # Add remaining part
        remaining = query[start:].strip()
        if remaining and len(remaining) > 5:
            sub_queries.append(remaining)
        
        # If we couldn't split properly, check for obvious multi-intent patterns
        if len(sub_queries) < 2:
            # Check for "X and Y" or "X e Y" patterns
            simple_split = re.split(r'\s+e\s+|\s+and\s+', query, 1)
            if len(simple_split) == 2 and len(simple_split[0].strip()) > 5 and len(simple_split[1].strip()) > 5:
                return simple_split
        
        return sub_queries if sub_queries else [query]
    
    def _handle_multi_intent(self, sub_queries: List[str], user_id: Optional[str], lang: str) -> Dict:
        """Handle multi-intent queries by processing each sub-query."""
        responses = []
        agents_used = []
        total_confidence = 0
        
        for sub_query in sub_queries:
            # Pass the language down in the recursive call
            result = self.route_query(sub_query, user_id, lang=lang)
            responses.append(result.get("answer", ""))
            agents_used.append(result.get("agent_used", "unknown"))
            total_confidence += result.get("confidence", 0)
        
        # Combine responses (localized header and item label)
        header = "🎯 **Answers to your questions:**" if lang and lang.split('-')[0] == "en" else "🎯 **Resposta para suas perguntas:**"
        item_label = "About" if lang and lang.split('-')[0] == "en" else "Sobre"
        combined_answer = f"{header}\n\n"
        for i, (query, response) in enumerate(zip(sub_queries, responses)):
            combined_answer += f"**{i+1}. {item_label}: {query}**\n{response}\n\n"
        
        avg_confidence = total_confidence / len(sub_queries) if sub_queries else 0
        
        return {
            "answer": combined_answer.strip(),
            "agent_used": "router",
            "intent": "multi_intent",
            "confidence": avg_confidence,
            "lang": lang,
            "sub_queries": sub_queries,
            "agents_used": agents_used
        }
    
    def get_agent_capabilities(self) -> Dict:
        """Get capabilities of available agents."""
        return {
            "support": {
                "description": "Assistente de suporte para questões de conta e transações",
                "capabilities": [
                    "Consultar saldo e dados da conta",
                    "Ver histórico de transações",
                    "Criar tickets de suporte",
                    "Ajuda com problemas de conta"
                ]
            },
            "knowledge": {
                "description": "Assistente de conhecimento para informações sobre produtos e serviços",
                "capabilities": [
                    "Informações sobre produtos InfinitePay",
                    "Explicar funcionalidades e taxas",
                    "Ajuda com como começar",
                    "Perguntas gerais sobre serviços"
                ]
            }
        }