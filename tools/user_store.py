"""Mock user data store and support tools."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class UserStore:
    """Mock user data store with support tools."""
    
    def __init__(self, data_path: str = None):
        if data_path is None:
            # Try to find the default path relative to different locations
            possible_paths = [
                "data/mock/users.json",
                "./data/mock/users.json", 
                "../data/mock/users.json",
            ]
            for path in possible_paths:
                if Path(path).exists():
                    data_path = path
                    break
            else:
                # Use the default path and let it fail gracefully
                data_path = "data/mock/users.json"
        
        self.data_path = Path(data_path)
        self.data = self._load_data()
    
    def _load_data(self) -> Dict:
        """Load mock data from JSON file."""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Mock data file not found: {self.data_path}")
            return {"users": [], "transactions": [], "support_tickets": []}
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in mock data file: {self.data_path}")
            return {"users": [], "transactions": [], "support_tickets": []}
    
    def get_user_by_id(self, user_id: str) -> Optional[Dict]:
        """Get user details by ID."""
        for user in self.data.get("users", []):
            if user.get("id") == user_id:
                return user
        return None
    
    def get_user_transactions(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Get recent transactions for a user."""
        transactions = [
            txn for txn in self.data.get("transactions", [])
            if txn.get("user_id") == user_id
        ]
        
        # Sort by date (newest first)
        transactions.sort(
            key=lambda x: x.get("created_at", ""),
            reverse=True
        )
        
        return transactions[:limit]
    
    def get_user_support_tickets(self, user_id: str) -> List[Dict]:
        """Get support tickets for a user."""
        return [
            ticket for ticket in self.data.get("support_tickets", [])
            if ticket.get("user_id") == user_id
        ]
    
    def create_support_ticket(self, user_id: str, subject: str, description: str) -> Optional[Dict]:
        """Create a new support ticket."""
        # Check if user exists
        user = self.get_user_by_id(user_id)
        if not user:
            return None
        
        # Generate new ticket ID
        existing_tickets = self.data.get("support_tickets", [])
        max_id = 0
        for ticket in existing_tickets:
            ticket_id = ticket.get("id", "")
            if ticket_id.startswith("ticket"):
                try:
                    num = int(ticket_id[6:])  # Remove "ticket" prefix
                    max_id = max(max_id, num)
                except ValueError:
                    continue
        
        new_ticket_id = f"ticket{str(max_id + 1).zfill(3)}"
        
        new_ticket = {
            "id": new_ticket_id,
            "user_id": user_id,
            "subject": subject,
            "description": description,
            "status": "open",
            "priority": "medium",  # Default priority
            "created_at": datetime.now().isoformat()
        }
        
        # Add to data
        if "support_tickets" not in self.data:
            self.data["support_tickets"] = []
        
        self.data["support_tickets"].append(new_ticket)
        
        # Save to file (for persistence)
        try:
            with open(self.data_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save mock data: {e}")
        
        return new_ticket


# Tool functions for the Support Agent
def get_account_details(user_id: str) -> str:
    """Get account details for a user.
    
    Args:
        user_id: The user ID
        
    Returns:
        Formatted account details string
    """
    store = UserStore()
    user = store.get_user_by_id(user_id)
    
    if not user:
        return f"❌ Usuário {user_id} não encontrado."
    
    # Format balance
    balance = user.get("balance", 0)
    balance_str = f"R$ {balance:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    
    # Format status
    status = user.get("status", "unknown")
    status_emoji = {
        "active": "✅",
        "suspended": "⚠️",
        "inactive": "❌"
    }.get(status, "❓")
    
    # Format account type
    account_type = user.get("account_type", "unknown")
    account_type_pt = {
        "personal": "Pessoal",
        "business": "Empresarial"
    }.get(account_type, account_type)
    
    response = f"""📋 **Dados da Conta**

**Nome:** {user.get('name', 'N/A')}
**Email:** {user.get('email', 'N/A')}
**Telefone:** {user.get('phone', 'N/A')}
**Tipo de Conta:** {account_type_pt}
**Saldo Atual:** {balance_str}
**Status:** {status_emoji} {status.title()}
**Data de Criação:** {user.get('created_at', 'N/A')}"""
    
    return response


def get_recent_transactions(user_id: str, limit: int = 5) -> str:
    """Get recent transactions for a user.
    
    Args:
        user_id: The user ID
        limit: Maximum number of transactions to return
        
    Returns:
        Formatted transactions string
    """
    store = UserStore()
    user = store.get_user_by_id(user_id)
    
    if not user:
        return f"❌ Usuário {user_id} não encontrado."
    
    transactions = store.get_user_transactions(user_id, limit)
    
    if not transactions:
        return "📊 **Transações Recentes**\n\nNenhuma transação encontrada."
    
    response = "📊 **Transações Recentes**\n\n"
    
    for txn in transactions:
        # Format amount
        amount = txn.get("amount", 0)
        amount_str = f"R$ {amount:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        
        # Format status
        status = txn.get("status", "unknown")
        status_emoji = {
            "completed": "✅",
            "pending": "⏳",
            "failed": "❌"
        }.get(status, "❓")
        
        # Format type
        txn_type = txn.get("type", "unknown")
        type_pt = {
            "payment": "Pagamento",
            "withdrawal": "Saque",
            "deposit": "Depósito"
        }.get(txn_type, txn_type)
        
        response += f"**{type_pt}** - {amount_str}\n"
        response += f"Descrição: {txn.get('description', 'N/A')}\n"
        response += f"Status: {status_emoji} {status.title()}\n"
        response += f"Data: {txn.get('created_at', 'N/A')}\n"
        response += "-" * 20 + "\n\n"
    
    return response.strip()


def open_support_ticket(user_id: str, subject: str, description: str) -> str:
    """Open a new support ticket.
    
    Args:
        user_id: The user ID
        subject: Ticket subject
        description: Ticket description
        
    Returns:
        Confirmation message
    """
    store = UserStore()
    ticket = store.create_support_ticket(user_id, subject, description)
    
    if not ticket:
        return f"❌ Não foi possível criar o ticket. Usuário {user_id} não encontrado."
    
    return f"""✅ **Ticket Criado com Sucesso!**

**ID do Ticket:** {ticket['id']}
**Assunto:** {ticket['subject']}
**Status:** {ticket['status'].title()}

Nossa equipe de suporte entrará em contato em breve.
Obrigado por entrar em contato com o InfinitePay!"""


# Tool metadata for agent selection
TOOL_METADATA = {
    "get_account_details": {
        "name": "get_account_details",
        "description": "Obter detalhes da conta do usuário",
        "keywords": [
            # Portuguese
            "conta", "saldo", "dados", "informações", "cadastro", "perfil", "login", "entrar",
            "acessar", "acesso", "minha conta", "meus dados", "dados cadastrais",
            # English
            "account", "balance", "details", "information", "profile", "login", "sign in",
            "access", "my account", "my data", "account details", "can't login", "cannot access"
        ],
        "requires_user_id": True
    },
    "get_recent_transactions": {
        "name": "get_recent_transactions",
        "description": "Obter transações recentes do usuário",
        "keywords": [
            # Portuguese
            "transações", "extrato", "pagamentos", "saques", "histórico", "movimentações",
            "transferências", "transações recentes", "extrato bancário",
            # English
            "transactions", "statement", "payments", "withdrawals", "history", "movements",
            "transfers", "recent transactions", "bank statement", "transaction history"
        ],
        "requires_user_id": True
    },
    "open_support_ticket": {
        "name": "open_support_ticket",
        "description": "Abrir novo ticket de suporte",
        "keywords": [
            # Portuguese
            "suporte", "ajuda", "problema", "ticket", "reclamação", "assistência", "atendimento",
            "preciso de ajuda", "tenho um problema", "falar com suporte",
            # English
            "support", "help", "issue", "problem", "ticket", "complaint", "assistance",
            "need help", "i have a problem", "contact support", "speak with support"
        ],
        "requires_user_id": True
    }
}


def get_tool_suggestions(query: str) -> List[str]:
    """Suggest tools based on query keywords."""
    query_lower = query.lower()
    suggestions = []
    
    for tool_name, metadata in TOOL_METADATA.items():
        for keyword in metadata["keywords"]:
            if keyword in query_lower:
                suggestions.append(tool_name)
                break
    
    return suggestions