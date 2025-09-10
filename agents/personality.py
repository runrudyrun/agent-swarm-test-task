"""Personality layer for final answer tone adjustment."""

import logging
import os
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class PersonalityLayer:
    """Optional personality layer to adjust tone of responses."""
    
    def __init__(self):
        self.enabled = os.getenv("PERSONALITY", "on").lower() == "on"
        self.locale = os.getenv("LOCALE", "pt-BR")
        
        # Tone guidelines
        self.tone_guidelines = {
            "friendly": True,
            "empathetic": True,
            "concise": True,
            "professional": True
        }
        
        # Common adjustments
        self.adjustments = {
            # Make more friendly
            "Olá": "Oi! 👋",
            "Prezado": "Oi",
            "Senhor(a)": "Você",
            
            # Add empathy
            "não consigo": "não conseguir",
            "problema": "situação",
            "erro": "dificuldade",
            
            # Add encouragement
            "entrar em contato": "entrar em contato - estamos aqui para ajudar! 💪",
            "aguardar": "aguardar - logo retornaremos",
        }
    
    def adjust_response(self, response: str, context: Optional[Dict] = None) -> str:
        """Adjust response tone if personality is enabled."""
        if not self.enabled:
            return response
        
        try:
            # Apply basic adjustments
            adjusted = self._apply_adjustments(response)
            
            # Add contextual improvements
            if context:
                adjusted = self._add_contextual_tone(adjusted, context)
            
            # Ensure proper formatting
            adjusted = self._format_response(adjusted)
            
            return adjusted
            
        except Exception as e:
            logger.error(f"Error adjusting response tone: {e}")
            return response  # Return original on error
    
    def _apply_adjustments(self, text: str) -> str:
        """Apply basic tone adjustments."""
        adjusted = text
        
        for old, new in self.adjustments.items():
            adjusted = adjusted.replace(old, new)
        
        # Add friendly closings if not present
        if not any(closing in adjusted for closing in ["Obrigado", "Atenciosamente", "Um abraço"]):
            adjusted += "\n\nConte comigo! 😊"
        
        return adjusted
    
    def _add_contextual_tone(self, text: str, context: Dict) -> str:
        """Add contextual tone improvements."""
        agent_used = context.get("agent_used", "")
        confidence = context.get("confidence", 1.0)
        
        # Add agent-specific tone
        if agent_used == "support":
            if "problema" in text.lower() or "erro" in text.lower():
                text = "🤝 Entendo sua preocupação. " + text
        
        elif agent_used == "knowledge":
            if confidence < 0.5:
                text = "🤔 Deixe-me verificar isso para você. " + text
        
        # Add confidence-based adjustments
        if confidence < 0.3:
            text += "\n\nSe precisar de mais detalhes, posso pesquisar mais informações para você! 🔍"
        
        return text
    
    def _format_response(self, text: str) -> str:
        """Ensure proper formatting."""
        # Remove excessive whitespace
        lines = text.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                formatted_lines.append(line)
        
        # Join with appropriate spacing
        result = '\n'.join(formatted_lines)
        
        # Ensure proper punctuation
        if result and not result[-1] in '.!?':
            result += '.'
        
        return result
    
    def is_enabled(self) -> bool:
        """Check if personality layer is enabled."""
        return self.enabled
    
    def get_tone_guidelines(self) -> Dict:
        """Get current tone guidelines."""
        return self.tone_guidelines.copy()