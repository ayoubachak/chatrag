from typing import List, Dict, Any, Optional, Union, AsyncGenerator
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
                        "top_p": 0.9,
                        "repetition_penalty": 1.1,
                        "pad_token_id": self.tokenizer.eos_token_id
                    }
                else:
                    # GPU parameters (can be more aggressive)
                    generation_config = {
                        "max_new_tokens": max_tokens,
                        "temperature": temperature,
                        "do_sample": temperature > 0,
                        "top_p": 0.9,
                        "repetition_penalty": 1.1,
                        "pad_token_id": self.tokenizer.eos_token_id
                    }
                
                # Log generation start
                start_time = time.time()
                self.logger.info(f"Starting generation with {generation_config['max_new_tokens']} max tokens")
                
                # Generate the response
                outputs = self.model.generate(
                    **inputs,
                    **generation_config
                )
                
                # Log generation time
                generation_time = time.time() - start_time
                self.logger.info(f"Generation completed in {generation_time:.2f} seconds")
                
                # Decode the response
                response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                
                return response
                
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}", exc_info=True)
            return f"Error generating response: {str(e)}"
            
    async def generate_stream(self, 
                           prompt: str, 
                           system_prompt: Optional[str] = None,
                           temperature: float = 0.7,
                           max_tokens: int = 1024,
                           context: Optional[List[Dict[str, Any]]] = None) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response using the local model.
        """
        # Ensure model is loaded
        if not await self.load_model():
            yield "Error: Failed to load model. Please check logs for details."
            return
        
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
            
            self.logger.debug(f"Formatted prompt for streaming: {prompt_text[:200]}...")
            
            # Tokenize the prompt
            inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
            
            # Check if we're running on CPU or GPU
            is_cpu = self.model.device.type == "cpu"
            
            # Set generation parameters based on device
            if is_cpu:
                # CPU-optimized parameters (more conservative)
                generation_config = {
                    "max_new_tokens": min(max_tokens, 512),  # Limit tokens on CPU
                    "temperature": temperature,
                    "do_sample": temperature > 0,
                    "top_p": 0.9,
                    "repetition_penalty": 1.1,
                    "pad_token_id": self.tokenizer.eos_token_id
                }
            else:
                # GPU parameters (can be more aggressive)
                generation_config = {
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "do_sample": temperature > 0,
                    "top_p": 0.9,
                    "repetition_penalty": 1.1,
                    "pad_token_id": self.tokenizer.eos_token_id
                }
            
            # Log generation start
            start_time = time.time()
            self.logger.info(f"Starting streaming generation with {generation_config['max_new_tokens']} max tokens")
            
            # Simpler approach: use the model's built-in streaming capability if available
            try:
                # Check if the model supports streaming
                if hasattr(self.model, 'generate_with_streaming') or hasattr(self.model.generate, 'with_streaming'):
                    # Use the model's built-in streaming capability
                    self.logger.info("Using model's built-in streaming capability")
                    
                    # Create a streamer
                    from transformers import TextIteratorStreamer
                    streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
                    
                    # Start generation in a separate thread
                    generation_kwargs = {
                        "input_ids": inputs.input_ids,
                        "streamer": streamer,
                        **generation_config
                    }
                    
                    import threading
                    thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
                    thread.start()
                    
                    # Yield from the streamer
                    for text in streamer:
                        self.logger.debug(f"Streaming chunk: {text}")
                        yield text
                        await asyncio.sleep(0.01)
                    
                    # Wait for the thread to finish
                    thread.join()
                    
                    # Log generation time
                    generation_time = time.time() - start_time
                    self.logger.info(f"Streaming generation completed in {generation_time:.2f} seconds")
                    return
            except Exception as e:
                self.logger.warning(f"Built-in streaming failed, falling back to token-by-token: {str(e)}")
                # Fall back to token-by-token generation
            
            # Stream the response using token-by-token generation
            input_length = inputs.input_ids.shape[1]
            generated_tokens = []
            
            with torch.no_grad():
                # Initial input
                current_input = inputs.input_ids
                past_key_values = None
                
                for _ in range(generation_config["max_new_tokens"]):
                    # Generate next token
                    outputs = self.model(
                        input_ids=current_input if past_key_values is None else current_input[:, -1:],
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    
                    # Get logits and past key values
                    logits = outputs.logits
                    past_key_values = outputs.past_key_values
                    
                    # Apply temperature if needed
                    if generation_config["do_sample"] and generation_config["temperature"] > 0:
                        logits = logits / generation_config["temperature"]
                    
                    # Apply top_p sampling if needed
                    if generation_config["do_sample"] and generation_config.get("top_p", 1.0) < 1.0:
                        sorted_logits, sorted_indices = torch.sort(logits[:, -1, :], descending=True)
                        cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > generation_config["top_p"]
                        # Shift the indices to the right to keep also the first token above the threshold
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        # Scatter sorted tensors to original indexing
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        logits[:, -1, :][indices_to_remove] = -float("Inf")
                    
                    # Apply repetition penalty
                    if generation_config.get("repetition_penalty", 1.0) > 1.0:
                        for i in range(current_input.shape[0]):
                            for token_idx in set(current_input[i].tolist()):
                                logits[i, -1, token_idx] /= generation_config["repetition_penalty"]
                    
                    # Sample next token
                    if generation_config["do_sample"]:
                        probs = torch.nn.functional.softmax(logits[:, -1, :], dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    
                    # Check if we've reached the end token
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
                    
                    # Add token to generated sequence
                    generated_tokens.append(next_token.item())
                    
                    # Update input for next iteration
                    current_input = torch.cat([current_input, next_token], dim=1)
                    
                    # Decode the new token and yield it
                    new_token_text = self.tokenizer.decode([next_token.item()], skip_special_tokens=True)
                    if new_token_text:  # Only yield non-empty text
                        self.logger.debug(f"Streaming token: {new_token_text}")
                        yield new_token_text
                        # Small delay to avoid overwhelming the client
                        await asyncio.sleep(0.01)
            
            # Log generation time
            generation_time = time.time() - start_time
            self.logger.info(f"Streaming generation completed in {generation_time:.2f} seconds")
                
        except Exception as e:
            self.logger.error(f"Error in streaming generation: {str(e)}", exc_info=True)
            yield f"Error in streaming generation: {str(e)}"

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
                        "top_p": 0.9,
                        "repetition_penalty": 1.1,
                        "pad_token_id": self.tokenizer.eos_token_id
                    }
                else:
                    # GPU parameters (can be more aggressive)
                    generation_config = {
                        "max_new_tokens": max_tokens,
                        "temperature": temperature,
                        "do_sample": temperature > 0,
                        "top_p": 0.9,
                        "repetition_penalty": 1.1,
                        "pad_token_id": self.tokenizer.eos_token_id
                    }
                
                # Log generation start
                start_time = time.time()
                self.logger.info(f"Starting RAG generation with {generation_config['max_new_tokens']} max tokens")
                
                # Generate the response
                outputs = self.model.generate(
                    **inputs,
                    **generation_config
                )
                
                # Log generation time
                generation_time = time.time() - start_time
                self.logger.info(f"RAG generation completed in {generation_time:.2f} seconds")
                
                # Decode the response
                response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                
                return response
                
        except Exception as e:
            self.logger.error(f"Error generating RAG response: {str(e)}", exc_info=True)
            return f"Error generating RAG response: {str(e)}"
            
    async def generate_with_rag_stream(self,
                                    prompt: str,
                                    documents: List[str],
                                    system_prompt: Optional[str] = None,
                                    temperature: float = 0.7,
                                    max_tokens: int = 1024,
                                    context: Optional[List[Dict[str, Any]]] = None) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response with RAG context using the local model.
        """
        # Ensure model is loaded
        if not await self.load_model():
            yield "Error: Failed to load model. Please check logs for details."
            return
        
        try:
            # Format the messages using our utility function
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
            
            # Convert messages to the format expected by the model
            prompt_text = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            self.logger.debug(f"Formatted RAG prompt for streaming: {prompt_text[:200]}...")
            
            # Tokenize the prompt
            inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
            
            # Check if we're running on CPU or GPU
            is_cpu = self.model.device.type == "cpu"
            
            # Set generation parameters based on device
            if is_cpu:
                # CPU-optimized parameters (more conservative)
                generation_config = {
                    "max_new_tokens": min(max_tokens, 512),  # Limit tokens on CPU
                    "temperature": temperature,
                    "do_sample": temperature > 0,
                    "top_p": 0.9,
                    "repetition_penalty": 1.1,
                    "pad_token_id": self.tokenizer.eos_token_id
                }
            else:
                # GPU parameters (can be more aggressive)
                generation_config = {
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "do_sample": temperature > 0,
                    "top_p": 0.9,
                    "repetition_penalty": 1.1,
                    "pad_token_id": self.tokenizer.eos_token_id
                }
            
            # Log generation start
            start_time = time.time()
            self.logger.info(f"Starting streaming RAG generation with {generation_config['max_new_tokens']} max tokens")
            
            # Simpler approach: use the model's built-in streaming capability if available
            try:
                # Check if the model supports streaming
                if hasattr(self.model, 'generate_with_streaming') or hasattr(self.model.generate, 'with_streaming'):
                    # Use the model's built-in streaming capability
                    self.logger.info("Using model's built-in streaming capability for RAG")
                    
                    # Create a streamer
                    from transformers import TextIteratorStreamer
                    streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
                    
                    # Start generation in a separate thread
                    generation_kwargs = {
                        "input_ids": inputs.input_ids,
                        "streamer": streamer,
                        **generation_config
                    }
                    
                    import threading
                    thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
                    thread.start()
                    
                    # Yield from the streamer
                    for text in streamer:
                        self.logger.debug(f"Streaming RAG chunk: {text}")
                        yield text
                        await asyncio.sleep(0.01)
                    
                    # Wait for the thread to finish
                    thread.join()
                    
                    # Log generation time
                    generation_time = time.time() - start_time
                    self.logger.info(f"Streaming RAG generation completed in {generation_time:.2f} seconds")
                    return
            except Exception as e:
                self.logger.warning(f"Built-in streaming failed for RAG, falling back to token-by-token: {str(e)}")
                # Fall back to token-by-token generation
            
            # Stream the response using token-by-token generation
            input_length = inputs.input_ids.shape[1]
            generated_tokens = []
            
            with torch.no_grad():
                # Initial input
                current_input = inputs.input_ids
                past_key_values = None
                
                for _ in range(generation_config["max_new_tokens"]):
                    # Generate next token
                    outputs = self.model(
                        input_ids=current_input if past_key_values is None else current_input[:, -1:],
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    
                    # Get logits and past key values
                    logits = outputs.logits
                    past_key_values = outputs.past_key_values
                    
                    # Apply temperature if needed
                    if generation_config["do_sample"] and generation_config["temperature"] > 0:
                        logits = logits / generation_config["temperature"]
                    
                    # Apply top_p sampling if needed
                    if generation_config["do_sample"] and generation_config.get("top_p", 1.0) < 1.0:
                        sorted_logits, sorted_indices = torch.sort(logits[:, -1, :], descending=True)
                        cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > generation_config["top_p"]
                        # Shift the indices to the right to keep also the first token above the threshold
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        # Scatter sorted tensors to original indexing
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        logits[:, -1, :][indices_to_remove] = -float("Inf")
                    
                    # Apply repetition penalty
                    if generation_config.get("repetition_penalty", 1.0) > 1.0:
                        for i in range(current_input.shape[0]):
                            for token_idx in set(current_input[i].tolist()):
                                logits[i, -1, token_idx] /= generation_config["repetition_penalty"]
                    
                    # Sample next token
                    if generation_config["do_sample"]:
                        probs = torch.nn.functional.softmax(logits[:, -1, :], dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    
                    # Check if we've reached the end token
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
                    
                    # Add token to generated sequence
                    generated_tokens.append(next_token.item())
                    
                    # Update input for next iteration
                    current_input = torch.cat([current_input, next_token], dim=1)
                    
                    # Decode the new token and yield it
                    new_token_text = self.tokenizer.decode([next_token.item()], skip_special_tokens=True)
                    if new_token_text:  # Only yield non-empty text
                        self.logger.debug(f"Streaming RAG token: {new_token_text}")
                        yield new_token_text
                        # Small delay to avoid overwhelming the client
                        await asyncio.sleep(0.01)
            
            # Log generation time
            generation_time = time.time() - start_time
            self.logger.info(f"Streaming RAG generation completed in {generation_time:.2f} seconds")
                
        except Exception as e:
            self.logger.error(f"Error in streaming RAG generation: {str(e)}", exc_info=True)
            yield f"Error in streaming RAG generation: {str(e)}"