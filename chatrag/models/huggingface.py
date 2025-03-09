import os
import httpx
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
import asyncio
import json

from openai import OpenAI, APIError
from .base import BaseLanguageModel
from .token_manager import TokenManager
from .utils import format_chat_history

logger = logging.getLogger("huggingface_model")

class HuggingFaceModel(BaseLanguageModel):
    """
    Implementation of the BaseLanguageModel for OpenAI-compatible API endpoints.
    Supports token rotation and automatic retry with different tokens.
    """

    def __init__(self, model_id: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        """
        Initialize the HuggingFace model client.
        
        Args:
            model_id: The model ID to use
        """
        self.model_id = model_id
        self.base_url = os.environ.get("HF_API_BASE_URL", "https://api.openai.com/v1")
        self.max_context_length = 4000  # Maximum context length in tokens
        self.timeout = 60  # Timeout in seconds for API calls
        self.max_retries = 3  # Maximum number of retries with different tokens
        
        # Initialize token manager
        self.token_manager = TokenManager("huggingface")
        
        if not self.token_manager.has_tokens:
            logger.warning("No HuggingFace API tokens available. Model will not function properly.")
        else:
            logger.info(f"Initialized HuggingFace model with {self.token_manager.token_count} tokens")

    async def _create_client(self):
        """Create an OpenAI client with the current token."""
        token = self.token_manager.get_token()
        if not token:
            raise ValueError("No API tokens available")
            
        return OpenAI(api_key=token, base_url=self.base_url, timeout=self.timeout)

    async def generate(self, 
                 prompt: str, 
                 system_prompt: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: int = 1024,
                 context: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Generate text using the HuggingFace API with token rotation.
        """
        # Format messages using our utility function
        messages = format_chat_history(
            messages=context or [],
            system_prompt=system_prompt,
            current_prompt=prompt
        )
        
        # Try with multiple tokens if needed
        for attempt in range(self.max_retries):
            try:
                # Get a client with the current token
                token = self.token_manager.get_token()
                if not token:
                    return "Error: No API tokens available"
                    
                client = await self._create_client()
                
                # Call the API with a timeout
                async def call_api():
                    try:
                        return await client.chat.completions.create(
                            messages=messages,
                            model=self.model_id,
                            temperature=temperature,
                            max_tokens=max_tokens
                        )
                    except (TypeError, AttributeError):
                        # Fallback for older versions or if async is not supported
                        return client.chat.completions.create(
                            messages=messages,
                            model=self.model_id,
                            temperature=temperature,
                            max_tokens=max_tokens
                        )
                
                try:
                    chat_completion = await asyncio.wait_for(call_api(), timeout=self.timeout)
                    # Report success
                    self.token_manager.report_success(token)
                    return chat_completion.choices[0].message.content
                except asyncio.TimeoutError:
                    logger.error(f"API call timed out after {self.timeout} seconds")
                    self.token_manager.report_failure(token)
                    continue
                    
            except APIError as e:
                logger.error(f"API error with token {token[:5]}...: {str(e)}")
                self.token_manager.report_failure(token)
                
                # Check if we should retry
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying with a different token (attempt {attempt + 2}/{self.max_retries})")
                    continue
                else:
                    return f"Error: API request failed after {self.max_retries} attempts: {str(e)}"
                    
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}", exc_info=True)
                return f"Error: {str(e)}"
                
        return "Error: Failed to generate response after multiple attempts"
        
    async def generate_stream(self, 
                           prompt: str, 
                           system_prompt: Optional[str] = None,
                           temperature: float = 0.7,
                           max_tokens: int = 1024,
                           context: Optional[List[Dict[str, Any]]] = None) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response using the HuggingFace API with token rotation.
        """
        # Format messages using our utility function
        messages = format_chat_history(
            messages=context or [],
            system_prompt=system_prompt,
            current_prompt=prompt
        )
        
        # Try with multiple tokens if needed
        for attempt in range(self.max_retries):
            try:
                # Get a client with the current token
                token = self.token_manager.get_token()
                if not token:
                    yield "Error: No API tokens available"
                    return
                    
                client = await self._create_client()
                
                try:
                    # Use streaming API - don't await the stream creation
                    stream = client.chat.completions.create(
                        messages=messages,
                        model=self.model_id,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=True
                    )
                    
                    # Process the stream - iterate over it directly
                    for chunk in stream:
                        if (hasattr(chunk.choices[0], 'delta') and 
                            hasattr(chunk.choices[0].delta, 'content') and 
                            chunk.choices[0].delta.content is not None):
                            content = chunk.choices[0].delta.content
                            logger.debug(f"Streaming chunk: {content}")
                            yield content
                            # Small pause to allow for cooperative multitasking
                            await asyncio.sleep(0)
                    
                    # Report success
                    self.token_manager.report_success(token)
                    return
                    
                except (APIError, Exception) as e:
                    logger.error(f"API error with token {token[:5]}...: {str(e)}")
                    self.token_manager.report_failure(token)
                    
                    # Check if we should retry
                    if attempt < self.max_retries - 1:
                        logger.info(f"Retrying streaming with a different token (attempt {attempt + 2}/{self.max_retries})")
                        continue
                    else:
                        yield f"Error: API streaming request failed after {self.max_retries} attempts: {str(e)}"
                        return
                        
            except Exception as e:
                logger.error(f"Unexpected error in streaming: {str(e)}", exc_info=True)
                yield f"Error: {str(e)}"
                return
                
        yield "Error: Failed to generate streaming response after multiple attempts"

    async def generate_with_rag(self,
                              prompt: str,
                              documents: List[str],
                              system_prompt: Optional[str] = None,
                              temperature: float = 0.7,
                              max_tokens: int = 1024,
                              context: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Generate text with RAG context using the HuggingFace API with token rotation.
        """
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
        
        # Try with multiple tokens if needed
        for attempt in range(self.max_retries):
            try:
                # Get a client with the current token
                token = self.token_manager.get_token()
                if not token:
                    return "Error: No API tokens available"
                    
                client = await self._create_client()
                
                # Call the API with a timeout
                async def call_api():
                    try:
                        return await client.chat.completions.create(
                            messages=messages,
                            model=self.model_id,
                            temperature=temperature,
                            max_tokens=max_tokens
                        )
                    except (TypeError, AttributeError):
                        # Fallback for older versions or if async is not supported
                        return client.chat.completions.create(
                            messages=messages,
                            model=self.model_id,
                            temperature=temperature,
                            max_tokens=max_tokens
                        )
                
                try:
                    chat_completion = await asyncio.wait_for(call_api(), timeout=self.timeout)
                    # Report success
                    self.token_manager.report_success(token)
                    return chat_completion.choices[0].message.content
                except asyncio.TimeoutError:
                    logger.error(f"API call timed out after {self.timeout} seconds")
                    self.token_manager.report_failure(token)
                    continue
                    
            except APIError as e:
                logger.error(f"API error with token {token[:5]}...: {str(e)}")
                self.token_manager.report_failure(token)
                
                # Check if we should retry
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying with a different token (attempt {attempt + 2}/{self.max_retries})")
                    continue
                else:
                    return f"Error: API request failed after {self.max_retries} attempts: {str(e)}"
                    
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}", exc_info=True)
                return f"Error: {str(e)}"
                
        return "Error: Failed to generate response after multiple attempts"
        
    async def generate_with_rag_stream(self,
                                    prompt: str,
                                    documents: List[str],
                                    system_prompt: Optional[str] = None,
                                    temperature: float = 0.7,
                                    max_tokens: int = 1024,
                                    context: Optional[List[Dict[str, Any]]] = None) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response with RAG context using the HuggingFace API with token rotation.
        """
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
        
        # Try with multiple tokens if needed
        for attempt in range(self.max_retries):
            try:
                # Get a client with the current token
                token = self.token_manager.get_token()
                if not token:
                    yield "Error: No API tokens available"
                    return
                    
                client = await self._create_client()
                
                try:
                    # Use streaming API - don't await the stream creation
                    stream = client.chat.completions.create(
                        messages=messages,
                        model=self.model_id,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=True
                    )
                    
                    # Process the stream - iterate over it directly
                    for chunk in stream:
                        if (hasattr(chunk.choices[0], 'delta') and 
                            hasattr(chunk.choices[0].delta, 'content') and 
                            chunk.choices[0].delta.content is not None):
                            content = chunk.choices[0].delta.content
                            logger.debug(f"Streaming RAG chunk: {content}")
                            yield content
                            # Small pause to allow for cooperative multitasking
                            await asyncio.sleep(0)
                    
                    # Report success
                    self.token_manager.report_success(token)
                    return
                    
                except (APIError, Exception) as e:
                    logger.error(f"API error with token {token[:5]}...: {str(e)}")
                    self.token_manager.report_failure(token)
                    
                    # Check if we should retry
                    if attempt < self.max_retries - 1:
                        logger.info(f"Retrying streaming with a different token (attempt {attempt + 2}/{self.max_retries})")
                        continue
                    else:
                        yield f"Error: API streaming request failed after {self.max_retries} attempts: {str(e)}"
                        return
                        
            except Exception as e:
                logger.error(f"Unexpected error in streaming: {str(e)}", exc_info=True)
                yield f"Error: {str(e)}"
                return
                
        yield "Error: Failed to generate streaming response after multiple attempts"