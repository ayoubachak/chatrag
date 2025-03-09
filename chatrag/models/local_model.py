from typing import List, Dict, Any, Optional, Union
import logging
import torch
from .base import BaseLanguageModel
from .local_model_manager import ModelManager
import os
import asyncio
import time
from .utils import format_chat_history

logger = logging.getLogger("local_model")

class LocalModel(BaseLanguageModel):
    """
    Implementation of the BaseLanguageModel for local models.
    """

    def __init__(self, model_path: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        """
        Initialize the local model.
        
        Args:
            model_path: Path or HuggingFace model ID to load
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.model_manager = ModelManager()
        self.logger = logging.getLogger("local_model")
        
        # Log initialization
        self.logger.info(f"Initialized local model with path: {model_path}")
        
        # Show model size warning only once
        if "7B" in model_path and not any(x in model_path.lower() for x in ["gptq", "gguf", "4bit", "8bit"]):
            self.logger.warning(
                "Using a 7B parameter model which requires significant GPU memory. "
                "Consider using smaller models if experiencing memory issues."
            )
        
    async def load_model(self):
        """
        Load the model and tokenizer.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get or load model and tokenizer from the manager
            self.model, self.tokenizer = self.model_manager.get_model_and_tokenizer(self.model_path)
            return True
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}", exc_info=True)
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
            
            # Check if we're running on CPU or GPU
            is_cpu = self.model.device.type == "cpu"
            
            # Generate response
            with torch.no_grad():
                # Set generation parameters based on device
                if is_cpu:
                    # CPU-optimized parameters (more conservative)
                    generation_config = {
                        "max_new_tokens": min(max_tokens, 512),  # Limit tokens on CPU
                        "temperature": temperature,
                        "do_sample": temperature > 0,
                        "top_p": 0.95,
                        "top_k": 50,
                        "repetition_penalty": 1.1,
                        "pad_token_id": self.tokenizer.eos_token_id,
                        "use_cache": True,
                    }
                else:
                    # GPU-optimized parameters
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
                system_prompt=None,  # We'll handle the system prompt separately
                current_prompt=None  # We'll handle the prompt separately
            )
            
            # Create a RAG-specific system prompt if none provided
            rag_system_prompt = system_prompt or "You are a helpful assistant. Answer the question based on the provided context. If the context doesn't contain relevant information, say so."
                
            # Format the retrieved documents
            context_str = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(documents)])
            
            # Combine system prompt and RAG context
            enhanced_system_prompt = f"{rag_system_prompt}\n\nContext:\n{context_str}"
            
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
            
            # Convert messages to the format expected by the model
            prompt_text = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            self.logger.debug(f"Formatted RAG prompt: {prompt_text[:200]}...")
            
            # Tokenize the prompt
            inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
            
            # Check if we're running on CPU or GPU
            is_cpu = self.model.device.type == "cpu"
            
            # Generate response
            with torch.no_grad():
                # Set generation parameters based on device
                if is_cpu:
                    # CPU-optimized parameters (more conservative)
                    generation_config = {
                        "max_new_tokens": min(max_tokens, 512),  # Limit tokens on CPU
                        "temperature": temperature,
                        "do_sample": temperature > 0,
                        "top_p": 0.95,
                        "top_k": 50,
                        "repetition_penalty": 1.1,
                        "pad_token_id": self.tokenizer.eos_token_id,
                        "use_cache": True,
                    }
                else:
                    # GPU-optimized parameters
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