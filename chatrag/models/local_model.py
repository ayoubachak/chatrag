from typing import List, Dict, Any, Optional
import logging
import torch
from .base import BaseLanguageModel
from .local_model_manager import ModelManager
import os
import asyncio
import time
from .utils import format_chat_history

class LocalModel(BaseLanguageModel):
    """
    Optimized implementation of the BaseLanguageModel using a shared Model Manager.
    """
    
    def __init__(self, model_path: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        """
        Initialize the local model using the Model Manager.
        """
        self.model_path = model_path
        self.logger = logging.getLogger(__name__)
        self.model_manager = ModelManager()
        
        # Show model size warning only once
        if "7B" in model_path and not any(x in model_path.lower() for x in ["gptq", "gguf", "4bit", "8bit"]):
            self.logger.warning(
                "Using a 7B parameter model which requires significant GPU memory. "
                "Consider using smaller models if experiencing memory issues."
            )
        
    async def load_model(self):
        """
        Load the model and tokenizer through the model manager.
        """
        try:
            # Get or load model and tokenizer from the manager
            self.model, self.tokenizer = self.model_manager.get_model_and_tokenizer(self.model_path)
            return True
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            return False

    async def generate(self, 
                      prompt: str, 
                      system_prompt: Optional[str] = None,
                      temperature: float = 0.7,
                      max_tokens: int = 1024,
                      context: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Generate text using the local model.
        """
        # Ensure model is loaded
        if not await self.load_model():
            return "Error: Failed to load model. Please check logs for details."
        
        try:
            # Format the messages using our utility function
            messages = format_chat_history(
                messages=context or [], 
                system_prompt=system_prompt,
                current_prompt=prompt
            )
            
            # Convert messages to the format expected by the model
            prompt_text = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            self.logger.debug(f"Formatted prompt: {prompt_text[:200]}...")
            
            # Tokenize the prompt
            inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
            
            # Generate response
            with torch.no_grad():
                # Set generation parameters
                generation_config = {
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "do_sample": temperature > 0,
                    "top_p": 0.95,
                    "top_k": 50,
                    "repetition_penalty": 1.1,
                }
                
                # Generate tokens
                start_time = time.time()
                output = self.model.generate(
                    **inputs,
                    **generation_config
                )
                end_time = time.time()
                
                self.logger.info(f"Generation took {end_time - start_time:.2f} seconds")
                
            # Decode the output
            generated_text = self.tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            return generated_text.strip()
            
        except Exception as e:
            self.logger.error(f"Error generating text: {str(e)}", exc_info=True)
            return f"Error generating response: {str(e)}"

    async def generate_with_rag(self,
                              prompt: str,
                              documents: List[str],
                              system_prompt: Optional[str] = None,
                              temperature: float = 0.7,
                              max_tokens: int = 1024,
                              context: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Generate text with RAG context using the local model.
        """
        # Ensure model is loaded
        if not await self.load_model():
            return "Error: Failed to load model. Please check logs for details."
        
        try:
            # Format the messages using our utility function
            messages = format_chat_history(
                messages=context or [], 
                system_prompt=system_prompt,
                current_prompt=None  # We'll handle the prompt separately with RAG
            )
            
            # Create a RAG-specific system prompt if none provided
            if not system_prompt:
                rag_system_prompt = "You are a helpful assistant. Answer the question based on the provided context. If the context doesn't contain relevant information, say so."
            else:
                rag_system_prompt = system_prompt
                
            # If we don't have a system message yet, add one
            if not messages or messages[0]["role"] != "system":
                messages.insert(0, {"role": "system", "content": rag_system_prompt})
                
            # Format the retrieved documents
            context_str = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(documents)])
            
            # Add the context and prompt as the final user message
            final_prompt = f"Context:\n{context_str}\n\nQuestion: {prompt}"
            
            # Add the final prompt as a user message
            if messages and messages[-1]["role"] == "user":
                # If the last message is from the user, append to it
                messages[-1]["content"] += f"\n\n{final_prompt}"
            else:
                # Otherwise add as a new message
                messages.append({"role": "user", "content": final_prompt})
            
            # Convert messages to the format expected by the model
            prompt_text = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            self.logger.debug(f"Formatted RAG prompt: {prompt_text[:200]}...")
            
            # Tokenize the prompt
            inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
            
            # Generate response
            with torch.no_grad():
                # Set generation parameters
                generation_config = {
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "do_sample": temperature > 0,
                    "top_p": 0.95,
                    "top_k": 50,
                    "repetition_penalty": 1.1,
                }
                
                # Generate tokens
                start_time = time.time()
                output = self.model.generate(
                    **inputs,
                    **generation_config
                )
                end_time = time.time()
                
                self.logger.info(f"RAG generation took {end_time - start_time:.2f} seconds")
                
            # Decode the output
            generated_text = self.tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            return generated_text.strip()
            
        except Exception as e:
            self.logger.error(f"Error generating RAG text: {str(e)}", exc_info=True)
            return f"Error generating RAG response: {str(e)}"