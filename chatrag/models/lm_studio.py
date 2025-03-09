import os
import httpx
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
from .base import BaseLanguageModel
from openai import OpenAI
from .utils import format_chat_history
import asyncio

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
            
    async def generate_stream(self, 
                           prompt: str, 
                           system_prompt: Optional[str] = None,
                           temperature: float = 0.7,
                           max_tokens: int = 1024,
                           context: Optional[List[Dict[str, Any]]] = None) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response using the LM Studio API.
        """
        try:
            # Format messages using our utility function
            messages = format_chat_history(
                messages=context or [],
                system_prompt=system_prompt,
                current_prompt=prompt
            )
            
            logger.debug(f"Sending {len(messages)} messages to LM Studio API (streaming)")
            
            # Use streaming API - don't await the stream creation
            stream = self.client.chat.completions.create(
                messages=messages,
                model="local-model", # LM Studio doesn't need a specific model name
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            # Yield each chunk as it arrives - iterate over it directly
            for chunk in stream:
                if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    logger.debug(f"Streaming chunk: {content}")
                    yield content
                    # Small pause to allow for cooperative multitasking
                    await asyncio.sleep(0)
                    
        except Exception as e:
            logger.error(f"Error generating streaming response with LM Studio: {str(e)}", exc_info=True)
            yield f"Error generating streaming response with LM Studio: {str(e)}"

    async def generate_with_rag(self,
                              prompt: str,
                              documents: List[str],
                              system_prompt: Optional[str] = None,
                              temperature: float = 0.7,
                              max_tokens: int = 1024,
                              context: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Generate text with RAG context using the LM Studio API.
        """
        try:
            # Format messages using our utility function
            messages = format_chat_history(
                messages=context or [],
                system_prompt=system_prompt,
                current_prompt=prompt
            )
            
            # Add retrieved documents to the system prompt
            if system_prompt:
                system_message = messages[0]
                documents_text = "\n\n".join([f"Document: {doc}" for doc in documents])
                system_message["content"] = f"{system_message['content']}\n\nRelevant context:\n{documents_text}"
            else:
                # If no system prompt, add documents as a system message
                documents_text = "\n\n".join([f"Document: {doc}" for doc in documents])
                system_message = {
                    "role": "system",
                    "content": f"You are a helpful assistant. Use the following information to answer the user's question:\n\n{documents_text}"
                }
                messages.insert(0, system_message)
            
            logger.debug(f"Sending {len(messages)} messages to LM Studio API with RAG")
            
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
            
    async def generate_with_rag_stream(self,
                                    prompt: str,
                                    documents: List[str],
                                    system_prompt: Optional[str] = None,
                                    temperature: float = 0.7,
                                    max_tokens: int = 1024,
                                    context: Optional[List[Dict[str, Any]]] = None) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response with RAG context using the LM Studio API.
        """
        try:
            # Format messages using our utility function
            messages = format_chat_history(
                messages=context or [],
                system_prompt=system_prompt,
                current_prompt=prompt
            )
            
            # Add retrieved documents to the system prompt
            if system_prompt:
                system_message = messages[0]
                documents_text = "\n\n".join([f"Document: {doc}" for doc in documents])
                system_message["content"] = f"{system_message['content']}\n\nRelevant context:\n{documents_text}"
            else:
                # If no system prompt, add documents as a system message
                documents_text = "\n\n".join([f"Document: {doc}" for doc in documents])
                system_message = {
                    "role": "system",
                    "content": f"You are a helpful assistant. Use the following information to answer the user's question:\n\n{documents_text}"
                }
                messages.insert(0, system_message)
            
            logger.debug(f"Sending {len(messages)} messages to LM Studio API with RAG (streaming)")
            
            # Use streaming API - don't await the stream creation
            stream = self.client.chat.completions.create(
                messages=messages,
                model="local-model", # LM Studio doesn't need a specific model name
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            # Yield each chunk as it arrives - iterate over it directly
            for chunk in stream:
                if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    logger.debug(f"Streaming RAG chunk: {content}")
                    yield content
                    # Small pause to allow for cooperative multitasking
                    await asyncio.sleep(0)
                    
        except Exception as e:
            logger.error(f"Error generating streaming RAG response with LM Studio: {str(e)}", exc_info=True)
            yield f"Error generating streaming RAG response with LM Studio: {str(e)}"