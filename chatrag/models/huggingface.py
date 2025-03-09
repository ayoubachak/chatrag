import os
import httpx
import logging
from typing import List, Dict, Any, Optional
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
                    # Try with a different token
                    self.token_manager.rotate_token()
                    continue
                    
            except APIError as e:
                # Check if it's a rate limit or quota error
                if e.status_code == 429 or e.status_code == 402:
                    logger.warning(f"Token exhausted or rate limited: {str(e)}")
                    self.token_manager.report_failure(token, is_rate_limit=True)
                    # Try with a different token
                    self.token_manager.rotate_token()
                    continue
                else:
                    logger.error(f"API error: {str(e)}")
                    return f"Error: {str(e)}"
            except Exception as e:
                logger.error(f"Error in generate: {str(e)}", exc_info=True)
                return f"Error generating response: {str(e)}"
        
        # If we've exhausted all retries
        return "Error: Unable to generate response after multiple attempts. Please try again later."

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
            
            # Limit the size of each document to avoid context overflow
            max_doc_length = 1000  # characters per document
            truncated_docs = []
            
            for i, doc in enumerate(documents):
                if len(doc) > max_doc_length:
                    truncated_doc = doc[:max_doc_length] + "..."
                    truncated_docs.append(truncated_doc)
                else:
                    truncated_docs.append(doc)
            
            # Limit the number of documents to include
            max_docs = 3
            if len(truncated_docs) > max_docs:
                # Keep only the top documents
                truncated_docs = truncated_docs[:max_docs]
            
            # Build the RAG context
            rag_context = "The following information may be relevant to the question:\n\n"
            rag_context += "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(truncated_docs)])
            rag_context += "\n\nUsing the above information, answer the following question."
            
            # Combine system prompt and RAG context
            enhanced_system_prompt = f"{rag_system_prompt}\n\n{rag_context}"
            
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
                messages.append({"role": "user", "content": prompt})
            
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
                        # Try with a different token
                        self.token_manager.rotate_token()
                        continue
                        
                except APIError as e:
                    # Check if it's a rate limit or quota error
                    if e.status_code == 429 or e.status_code == 402:
                        logger.warning(f"Token exhausted or rate limited: {str(e)}")
                        self.token_manager.report_failure(token, is_rate_limit=True)
                        # Try with a different token
                        self.token_manager.rotate_token()
                        continue
                    else:
                        logger.error(f"API error: {str(e)}")
                        return f"Error: {str(e)}"
                except Exception as e:
                    logger.error(f"Error in generate_with_rag: {str(e)}", exc_info=True)
                    return f"Error generating response: {str(e)}"
            
            # If we've exhausted all retries
            return "Error: Unable to generate response after multiple attempts. Please try again later."
            
        except Exception as e:
            logger.error(f"Error in generate_with_rag: {str(e)}", exc_info=True)
            return f"Error generating response with RAG: {str(e)}"