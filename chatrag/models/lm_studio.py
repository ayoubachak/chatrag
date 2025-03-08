import os
import httpx
from typing import List, Dict, Any, Optional
from .base import BaseLanguageModel

class LMStudioModel(BaseLanguageModel):
    """
    Implementation of the BaseLanguageModel for LM Studio's local API.
    This uses the OpenAI-compatible API that LM Studio provides.
    """
    
    def __init__(self, api_base: str = "http://localhost:1234/v1"):
        """
        Initialize the LM Studio model client.
        
        Args:
            api_base: Base URL for the LM Studio API
        """
        self.api_base = api_base
        self.client = httpx.AsyncClient(timeout=60.0)

    async def generate(self, 
                      prompt: str, 
                      system_prompt: Optional[str] = None,
                      temperature: float = 0.7,
                      max_tokens: int = 1024,
                      context: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Generate text using the LM Studio API.
        """
        # Format the messages according to OpenAI's chat format
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        # Add previous context messages if provided
        if context:
            messages.extend(context)
            
        # Add the current prompt
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        try:
            response = await self.client.post(
                f"{self.api_base}/chat/completions", 
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            
            # Extract the generated text from the response
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                return "No response generated"
        except Exception as e:
            return f"Error generating response with LM Studio: {str(e)}"
            
    async def generate_with_rag(self,
                              prompt: str,
                              documents: List[str],
                              system_prompt: Optional[str] = None,
                              temperature: float = 0.7,
                              max_tokens: int = 1024,
                              context: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Generate a response with RAG-enhanced context.
        """
        # Prepare RAG-enhanced prompt with retrieved documents
        rag_context = "The following information may be relevant to the question:\n\n"
        rag_context += "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(documents)])
        rag_context += "\n\nUsing the above information, answer the following question."
        
        # If system prompt is provided, combine it with RAG context
        if system_prompt:
            enhanced_system_prompt = f"{system_prompt}\n\n{rag_context}"
        else:
            enhanced_system_prompt = rag_context
            
        # Use the regular generate function with the enhanced system prompt
        return await self.generate(
            prompt=prompt,
            system_prompt=enhanced_system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            context=context
        )