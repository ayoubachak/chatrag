import os
import httpx
import logging
from typing import List, Dict, Any, Optional
from .base import BaseLanguageModel
from openai import OpenAI
from .utils import format_chat_history

logger = logging.getLogger("lm_studio_model")

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
        self.client = OpenAI(base_url=self.api_base, api_key="not-needed")
        logger.info(f"Initialized LM Studio model with API base: {api_base}")

    async def generate(self, 
                      prompt: str, 
                      system_prompt: Optional[str] = None,
                      temperature: float = 0.7,
                      max_tokens: int = 1024,
                      context: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Generate text using the LM Studio API.
        """
        try:
            # Format messages using our utility function
            messages = format_chat_history(
                messages=context or [],
                system_prompt=system_prompt,
                current_prompt=prompt
            )
            
            logger.debug(f"Sending {len(messages)} messages to LM Studio API")
            
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
            
            response = chat_completion.choices[0].message.content
            logger.debug(f"Generated response: {response[:50]}...")
            return response
        except Exception as e:
            logger.error(f"Error generating response with LM Studio: {str(e)}", exc_info=True)
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
        try:
            # Format messages using our utility function
            messages = format_chat_history(
                messages=context or [],
                system_prompt=None,  # We'll handle the system prompt separately
                current_prompt=None  # We'll handle the prompt separately
            )
            
            # Create a RAG-specific system prompt
            rag_system_prompt = system_prompt or "You are a helpful assistant. Answer the question based on the provided context."
            
            # Format the retrieved documents
            context_str = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(documents)])
            
            # Combine system prompt and RAG context
            enhanced_system_prompt = f"{rag_system_prompt}\n\nContext:\n{context_str}\n\nAnswer the following question based on the provided context."
            
            # Add system message at the beginning
            if not messages or messages[0]["role"] != "system":
                messages.insert(0, {"role": "system", "content": enhanced_system_prompt})
            else:
                messages[0]["content"] = enhanced_system_prompt
                
            # Add the user prompt as the last message
            if messages and messages[-1]["role"] == "user":
                # If the last message is from the user, append to it
                messages[-1]["content"] += f"\n\nQuestion: {prompt}"
            else:
                # Otherwise add as a new message
                messages.append({"role": "user", "content": f"Question: {prompt}"})
            
            logger.debug(f"Sending {len(messages)} RAG-enhanced messages to LM Studio API")
            
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
            
            response = chat_completion.choices[0].message.content
            logger.debug(f"Generated RAG response: {response[:50]}...")
            return response
        except Exception as e:
            logger.error(f"Error generating RAG response with LM Studio: {str(e)}", exc_info=True)
            return f"Error generating RAG response with LM Studio: {str(e)}"