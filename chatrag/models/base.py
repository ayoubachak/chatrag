from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseLanguageModel(ABC):
    """
    Abstract base class for all language model implementations.
    This ensures a consistent interface regardless of the model source.
    """
    
    @abstractmethod
    async def generate(self, 
                      prompt: str, 
                      system_prompt: Optional[str] = None,
                      temperature: float = 0.7,
                      max_tokens: int = 1024,
                      context: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Generate a response based on the prompt and optional context.
        
        Args:
            prompt: The user's message to respond to
            system_prompt: Optional system prompt to guide the model
            temperature: Controls randomness (higher = more random)
            max_tokens: Maximum number of tokens to generate
            context: Optional list of previous messages for context
            
        Returns:
            The generated text response
        """
        pass
    
    @abstractmethod
    async def generate_with_rag(self,
                              prompt: str,
                              documents: List[str],
                              system_prompt: Optional[str] = None,
                              temperature: float = 0.7,
                              max_tokens: int = 1024,
                              context: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Generate a response with RAG-enhanced context.
        
        Args:
            prompt: The user's message to respond to
            documents: List of retrieved document snippets to include as context
            system_prompt: Optional system prompt to guide the model
            temperature: Controls randomness (higher = more random)
            max_tokens: Maximum number of tokens to generate
            context: Optional list of previous messages for context
            
        Returns:
            The generated text response
        """
        pass