from abc import ABC, abstractmethod
from typing import List, Dict, Any

class AIBackend(ABC):
    """Abstract base class for AI backends (Ollama, Claude, etc.)"""
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the AI backend and check availability"""
        pass
    
    @abstractmethod
    async def generate_response(self, system_prompt: str, messages: List[Dict[str, str]], 
                           max_tokens: int = 2000, temperature: float = 0) -> Dict[str, Any]:
        """Generate a response from the AI model"""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the current model name"""
        pass
    
    @abstractmethod
    async def set_model(self, model_name: str) -> bool:
        """Set a new model and check its availability"""
        pass