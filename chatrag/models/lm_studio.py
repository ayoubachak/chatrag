import os
import httpx
from typing import List, Dict, Any, Optional
from .base import BaseLanguageModel
from openai import OpenAI

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
        # Replace httpx client with OpenAI client
        self.client = OpenAI(base_url=self.api_base, api_key="not-needed")

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
            
        # Add previous context messages if provided, ensuring they follow the alternating pattern
        if context:
            # If we have context, we need to validate it follows the user/assistant alternating pattern
            # or construct a valid conversation history
            
            # Check if the context already has the proper alternating pattern
            valid_context = True
            expected_roles = ["user", "assistant"] * (len(context) // 2 + 1)
            
            for i, msg in enumerate(context):
                if i < len(expected_roles) and msg.get("role") != expected_roles[i]:
                    valid_context = False
                    break
                # Also check if role is missing or undefined
                if "role" not in msg or msg.get("role") is None:
                    valid_context = False
                    break
            
            if valid_context:
                messages.extend(context)
            else:
                # If context doesn't have proper alternating pattern, reconstruct it
                current_role = "user"
                for msg in context:
                    if "content" in msg and msg.get("content") is not None:
                        messages.append({"role": current_role, "content": msg.get("content", "")})
                        # Toggle between user and assistant
                        current_role = "assistant" if current_role == "user" else "user"
            
        # Add the current prompt
        messages.append({"role": "user", "content": prompt})
        
        try:
            # Use async version of the OpenAI client if available, otherwise use synchronous version
            try:
                # For newer versions of the OpenAI client that support async
                chat_completion = await self.client.chat.completions.create(
                    messages=messages,
                    model="local-model", # LM Studio doesn't need a specific model name
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            except (TypeError, AttributeError):
                # Fallback for older versions or if async is not supported
                chat_completion = self.client.chat.completions.create(
                    messages=messages,
                    model="local-model", # LM Studio doesn't need a specific model name
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            
            return chat_completion.choices[0].message.content
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