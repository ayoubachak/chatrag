import os
import httpx
from typing import List, Dict, Any, Optional
import asyncio

from openai import OpenAI
from .base import BaseLanguageModel

class HuggingFaceModelLegacy(BaseLanguageModel):
    """
    Implementation of the BaseLanguageModel for HuggingFace API.
    """

    def __init__(self, model_id: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        """
        Initialize the HuggingFace model client.
        
        Args:
            model_id: The model ID to use on HuggingFace
        """
        self.model_id = model_id
        self.api_token = os.environ.get("HF_API_TOKEN")
        self.api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        
        if not self.api_token:
            raise ValueError("HuggingFace API token not found. Please set the HF_API_TOKEN environment variable.")
            
        self.headers = {"Authorization": f"Bearer {self.api_token}"}
        self.client = httpx.AsyncClient(headers=self.headers, timeout=60.0)

    async def generate(self, 
                       prompt: str, 
                       system_prompt: Optional[str] = None,
                       temperature: float = 0.7,
                       max_tokens: int = 1024,
                       context: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Generate text using the HuggingFace API.
        """
        # Combine all parts into a single text prompt
        final_prompt = ""
        if system_prompt:
            final_prompt += system_prompt.strip() + "\n\n"
        
        if context:
            # Append previous context messages (assuming each message has 'role' and 'content')
            for message in context:
                role = message.get("role", "").capitalize()
                content = message.get("content", "")
                final_prompt += f"{role}: {content.strip()}\n"
            final_prompt += "\n"
        
        # Append the user prompt
        final_prompt += f"User: {prompt.strip()}\n"
        # Optionally, you can add a marker for the assistant's response.
        final_prompt += "Assistant:"

        payload = {
            "inputs": final_prompt,
            "parameters": {
                "temperature": temperature,
                "max_new_tokens": max_tokens,
                "return_full_text": False
            }
        }
        
        try:
            response = await self.client.post(self.api_url, json=payload)
            response.raise_for_status()
            result = response.json()
            
            # Extract the generated text from the response
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "")
            else:
                return result.get("generated_text", "")
        except Exception as e:
            return f"Error generating response: {str(e)}"
            
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
        # Prepare RAG-enhanced context by combining the retrieved documents
        rag_context = "The following information may be relevant to the question:\n\n"
        rag_context += "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(documents)])
        rag_context += "\n\nUsing the above information, answer the following question."
        
        # Combine system prompt and RAG context if a system prompt exists
        if system_prompt:
            enhanced_system_prompt = f"{system_prompt.strip()}\n\n{rag_context}"
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


class HuggingFaceModel(BaseLanguageModel):
    """
    Implementation of the BaseLanguageModel for OpenAI API.
    """

    def __init__(self, model_id: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        """
        Initialize the OpenAI model client.
        
        Args:
            model_id: The model ID to use on OpenAI API.
        """
        self.model_id = model_id
        self.api_key = os.environ.get("HF_API_TOKEN")
        self.base_url = os.environ.get("HF_API_BASE_URL", "https://api.openai.com/v1")
        self.max_context_length = 4000  # Maximum context length in tokens
        self.timeout = 60  # Timeout in seconds for API calls
        
        if not self.api_key:
            raise ValueError("OpenAI API token not found. Please set the HF_API_TOKEN environment variable.")
            
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

    async def generate(self, 
                 prompt: str, 
                 system_prompt: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: int = 1024,
                 context: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Generate text using the OpenAI API.
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt.strip()})
            
        # Add context messages if provided, ensuring they follow the alternating pattern
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
            
            if valid_context:
                messages.extend(context)
            else:
                # If context doesn't have proper alternating pattern, reconstruct it
                current_role = "user"
                for msg in context:
                    messages.append({"role": current_role, "content": msg.get("content", "")})
                    # Toggle between user and assistant
                    current_role = "assistant" if current_role == "user" else "user"
            
        # Add the current user prompt
        messages.append({"role": "user", "content": prompt.strip()})
        
        # Use async version of the OpenAI client if available, otherwise use synchronous version
        try:
            # Implement timeout using asyncio.wait_for
            async def call_api():
                try:
                    # For newer versions of the OpenAI client that support async
                    return await self.client.chat.completions.create(
                        messages=messages,
                        model=self.model_id,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                except (TypeError, AttributeError):
                    # Fallback for older versions or if async is not supported
                    return self.client.chat.completions.create(
                        messages=messages,
                        model=self.model_id,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
            
            # Call the API with a timeout
            try:
                chat_completion = await asyncio.wait_for(call_api(), timeout=self.timeout)
                return chat_completion.choices[0].message.content
            except asyncio.TimeoutError:
                import logging
                logging.error(f"API call to HuggingFace timed out after {self.timeout} seconds")
                return f"Error: The request to the model timed out after {self.timeout} seconds. Please try again with a simpler query or fewer documents."
                
        except Exception as e:
            import logging
            logging.error(f"Error in HuggingFaceModel.generate: {str(e)}", exc_info=True)
            return f"Error generating response: {str(e)}"

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
            
            if system_prompt:
                enhanced_system_prompt = f"{system_prompt.strip()}\n\n{rag_context}"
            else:
                enhanced_system_prompt = rag_context
            
            # Generate response with enhanced context
            response = await self.generate(
                prompt=prompt,
                system_prompt=enhanced_system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                context=context
            )
            
            # Ensure we have a valid string response
            if not response or not isinstance(response, str):
                return "Error: Invalid response format from model"
            
            return response.strip()
        except Exception as e:
            import logging
            logging.error(f"Error in HuggingFaceModel.generate_with_rag: {str(e)}", exc_info=True)
            return f"Error generating response with RAG: {str(e)}"