import os
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
import asyncio
import json

from .base import BaseLanguageModel
from .token_manager import TokenManager
from .utils import format_chat_history

logger = logging.getLogger("huggingface_model")

try:
    from openai import OpenAI, APIError
    OPENAI_CLIENT_AVAILABLE = True
except ImportError:
    OPENAI_CLIENT_AVAILABLE = False
    logger.warning("openai not available. Install with 'pip install openai'")

class HuggingFaceModel(BaseLanguageModel):
    """
    Implementation of the BaseLanguageModel for HuggingFace models using the OpenAI client.
    Ensures conversation roles properly alternate user/assistant/user/assistant.
    """

    def __init__(self, model_id: str = "google/gemma-2-2b-it"):
        """
        Initialize the HuggingFace model client.
        
        Args:
            model_id: The model ID to use
        """
        self.model_id = model_id
        self.router_url = "https://router.huggingface.co/hf-inference/v1"
        self.max_context_length = 4000  # Maximum context length in tokens
        self.timeout = 120  # Timeout in seconds for API calls
        self.max_retries = 3  # Maximum number of retries with different tokens
        
        # Initialize token manager
        self.token_manager = TokenManager("huggingface")
        
        if not self.token_manager.has_tokens:
            logger.warning("No HuggingFace API tokens available. Model will not function properly.")
        else:
            logger.info(f"Initialized HuggingFace model with {self.token_manager.token_count} tokens")
            
        # Check if OpenAI client is available
        if not OPENAI_CLIENT_AVAILABLE:
            logger.error("OpenAI library is not available. Install with 'pip install openai'")

    def _create_openai_client(self, token: str):
        """Create an OpenAI client configured for HF Inference API."""
        if not OPENAI_CLIENT_AVAILABLE:
            return None
            
        return OpenAI(
            base_url=self.router_url,
            api_key=token,
            timeout=self.timeout
        )

    def _ensure_alternating_roles(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Ensure the conversation roles alternate properly between user and assistant.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            List of formatted messages with proper role alternation
        """
        # Drop system messages entirely
        messages = [msg for msg in messages if msg["role"] != "system"]
        
        # If no messages, add a default user message
        if not messages:
            return [{"role": "user", "content": "Hello"}]
        
        # Make sure the first message is from user
        if messages[0]["role"] != "user":
            # Insert a generic user message
            messages.insert(0, {"role": "user", "content": "I need your assistance."})
            
        # Check message alternation and fix if needed
        fixed_messages = [messages[0]]  # Start with the first message (user)
        
        for i in range(1, len(messages)):
            prev_role = fixed_messages[-1]["role"]
            curr_role = messages[i]["role"]
            
            # If roles don't alternate properly, insert a padding message
            if prev_role == curr_role:
                if curr_role == "user":
                    # Insert assistant message between consecutive user messages
                    fixed_messages.append({
                        "role": "assistant", 
                        "content": "I understand. Please continue."
                    })
                else:  # curr_role == "assistant"
                    # Insert user message between consecutive assistant messages
                    fixed_messages.append({
                        "role": "user", 
                        "content": "Please continue."
                    })
            
            # Add the current message
            fixed_messages.append(messages[i])
        
        # Make sure we end with an assistant message if the last message is from user
        if fixed_messages[-1]["role"] == "user":
            # This shouldn't happen in normal usage, but just in case
            logger.warning("Last message is from user, which is unexpected in a completion scenario")
        
        return fixed_messages

    def _prepare_rag_context(self, documents: List[str]) -> str:
        """
        Format RAG documents into a context string.
        
        Args:
            documents: List of document strings
            
        Returns:
            Formatted context string
        """
        return "\n\n".join([f"Document: {doc}" for doc in documents])

    async def generate(self, 
                 prompt: str, 
                 system_prompt: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: int = 1024,
                 context: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Generate text using the HuggingFace chat completions API.
        """
        # Start with a simpler approach - just the user's prompt
        # This is more reliable with models that have strict format requirements
        messages = [{"role": "user", "content": prompt}]
        
        # If system prompt is provided, add it to the user message
        if system_prompt:
            messages[0]["content"] = f"{system_prompt}\n\n{prompt}"
            
        # If context is provided and not empty, use it instead, ensuring alternation
        if context and len(context) > 0:
            context_messages = format_chat_history(
                messages=context,
                current_prompt=prompt
            )
            messages = self._ensure_alternating_roles(context_messages)
        
        # Try with multiple tokens if needed
        for attempt in range(self.max_retries):
            try:
                # Get a token
                token = self.token_manager.get_token()
                if not token:
                    return "Error: No API tokens available"
                
                try:
                    # Create OpenAI client
                    client = self._create_openai_client(token)
                    if not client:
                        return "Error: OpenAI client not available"
                    
                    # Log the messages being sent
                    logger.debug(f"Sending messages: {json.dumps(messages)}")
                    
                    # Make the API call
                    response = await asyncio.wait_for(
                        asyncio.to_thread(
                            client.chat.completions.create,
                            model=self.model_id,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens
                        ),
                        timeout=self.timeout
                    )
                    
                    # Report success
                    self.token_manager.report_success(token)
                    return response.choices[0].message.content
                    
                except asyncio.TimeoutError:
                    logger.error(f"API call timed out after {self.timeout} seconds")
                    self.token_manager.report_failure(token)
                    continue
                    
            except Exception as e:
                logger.error(f"API error with token {token[:5]}...: {str(e)}", exc_info=True)
                self.token_manager.report_failure(token)
                
                # Check if we should retry
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying with a different token (attempt {attempt + 2}/{self.max_retries})")
                    continue
                else:
                    return f"Error: API request failed after {self.max_retries} attempts: {str(e)}"
                
        return "Error: Failed to generate response after multiple attempts"
        
    async def generate_stream(self, 
                           prompt: str, 
                           system_prompt: Optional[str] = None,
                           temperature: float = 0.7,
                           max_tokens: int = 1024,
                           context: Optional[List[Dict[str, Any]]] = None) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response using the HuggingFace chat completions API.
        """
        # Start with a simpler approach - just the user's prompt
        messages = [{"role": "user", "content": prompt}]
        
        # If system prompt is provided, add it to the user message
        if system_prompt:
            messages[0]["content"] = f"{system_prompt}\n\n{prompt}"
            
        # If context is provided and not empty, use it instead, ensuring alternation
        if context and len(context) > 0:
            context_messages = format_chat_history(
                messages=context,
                current_prompt=prompt
            )
            messages = self._ensure_alternating_roles(context_messages)
        
        # Try with multiple tokens if needed
        for attempt in range(self.max_retries):
            try:
                # Get a token
                token = self.token_manager.get_token()
                if not token:
                    yield "Error: No API tokens available"
                    return
                
                try:
                    # Create OpenAI client
                    client = self._create_openai_client(token)
                    if not client:
                        yield "Error: OpenAI client not available"
                        return
                    
                    # Log the messages being sent
                    logger.debug(f"Sending streaming messages: {json.dumps(messages)}")
                        
                    # Create a streaming response
                    stream = client.chat.completions.create(
                        model=self.model_id,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=True
                    )
                    
                    # Stream the response
                    for chunk in stream:
                        if (hasattr(chunk.choices[0], 'delta') and 
                            hasattr(chunk.choices[0].delta, 'content') and 
                            chunk.choices[0].delta.content):
                            content = chunk.choices[0].delta.content
                            yield content
                            await asyncio.sleep(0)  # Allow for cooperative multitasking
                    
                    # Report success
                    self.token_manager.report_success(token)
                    return
                    
                except Exception as e:
                    logger.error(f"Streaming error: {str(e)}", exc_info=True)
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
        Generate text with RAG context using the HuggingFace chat completions API.
        """
        # Format RAG context
        rag_context = self._prepare_rag_context(documents)
        
        # Start with a simple user message that includes the RAG context
        messages = [{
            "role": "user", 
            "content": f"I need information about the following: {prompt}\n\nHere is relevant information to help you:\n\n{rag_context}"
        }]
        
        # If system prompt is provided, incorporate it into the user message
        if system_prompt:
            messages[0]["content"] = f"{system_prompt}\n\n{messages[0]['content']}"
        
        # If context is provided and not empty, use it appropriately
        if context and len(context) > 0:
            # Add RAG context to the end of the last user message
            context_messages = format_chat_history(
                messages=context,
                current_prompt=f"{prompt}\n\nHere is relevant information:\n\n{rag_context}"
            )
            messages = self._ensure_alternating_roles(context_messages)
        
        # Try with multiple tokens if needed
        for attempt in range(self.max_retries):
            try:
                # Get a token
                token = self.token_manager.get_token()
                if not token:
                    return "Error: No API tokens available"
                
                try:
                    # Create OpenAI client
                    client = self._create_openai_client(token)
                    if not client:
                        return "Error: OpenAI client not available"
                    
                    # Log the messages being sent
                    logger.debug(f"Sending RAG messages: {json.dumps(messages)}")
                    
                    # Make the API call
                    response = await asyncio.wait_for(
                        asyncio.to_thread(
                            client.chat.completions.create,
                            model=self.model_id,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens
                        ),
                        timeout=self.timeout
                    )
                    
                    # Report success
                    self.token_manager.report_success(token)
                    return response.choices[0].message.content
                    
                except asyncio.TimeoutError:
                    logger.error(f"API call timed out after {self.timeout} seconds")
                    self.token_manager.report_failure(token)
                    continue
                    
            except Exception as e:
                logger.error(f"API error with token {token[:5]}...: {str(e)}", exc_info=True)
                self.token_manager.report_failure(token)
                
                # Check if we should retry
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying with a different token (attempt {attempt + 2}/{self.max_retries})")
                    continue
                else:
                    return f"Error: API request failed after {self.max_retries} attempts: {str(e)}"
                
        return "Error: Failed to generate response after multiple attempts"
        
    async def generate_with_rag_stream(self,
                                    prompt: str,
                                    documents: List[str],
                                    system_prompt: Optional[str] = None,
                                    temperature: float = 0.7,
                                    max_tokens: int = 1024,
                                    context: Optional[List[Dict[str, Any]]] = None) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response with RAG context using the HuggingFace chat completions API.
        """
        # Format RAG context
        rag_context = self._prepare_rag_context(documents)
        
        # Start with a simple user message that includes the RAG context
        messages = [{
            "role": "user", 
            "content": f"I need information about the following: {prompt}\n\nHere is relevant information to help you:\n\n{rag_context}"
        }]
        
        # If system prompt is provided, incorporate it into the user message
        if system_prompt:
            messages[0]["content"] = f"{system_prompt}\n\n{messages[0]['content']}"
        
        # If context is provided and not empty, use it appropriately
        if context and len(context) > 0:
            # Add RAG context to the end of the last user message
            context_messages = format_chat_history(
                messages=context,
                current_prompt=f"{prompt}\n\nHere is relevant information:\n\n{rag_context}"
            )
            messages = self._ensure_alternating_roles(context_messages)
        
        # Try with multiple tokens if needed
        for attempt in range(self.max_retries):
            try:
                # Get a token
                token = self.token_manager.get_token()
                if not token:
                    yield "Error: No API tokens available"
                    return
                
                try:
                    # Create OpenAI client
                    client = self._create_openai_client(token)
                    if not client:
                        yield "Error: OpenAI client not available"
                        return
                        
                    # Log the messages being sent
                    logger.debug(f"Sending RAG streaming messages: {json.dumps(messages)}")
                    
                    # Create a streaming response
                    stream = client.chat.completions.create(
                        model=self.model_id,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=True
                    )
                    
                    # Stream the response
                    for chunk in stream:
                        if (hasattr(chunk.choices[0], 'delta') and 
                            hasattr(chunk.choices[0].delta, 'content') and 
                            chunk.choices[0].delta.content):
                            content = chunk.choices[0].delta.content
                            yield content
                            await asyncio.sleep(0)  # Allow for cooperative multitasking
                    
                    # Report success
                    self.token_manager.report_success(token)
                    return
                    
                except Exception as e:
                    logger.error(f"Streaming error: {str(e)}", exc_info=True)
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