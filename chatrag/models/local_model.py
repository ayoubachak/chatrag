from typing import List, Dict, Any, Optional
import logging
import torch
from .base import BaseLanguageModel
from .local_model_manager import ModelManager

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
            # Format the messages according to the model's expected format
            messages = []
            
            # Add system prompt if provided
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
                
            # Add previous context messages if provided
            if context:
                # Validate and process context to ensure proper role alternation
                valid_messages = []
                expected_roles = ["user", "assistant"] * (len(context) // 2 + 1)
                
                # If we already have a system message, adjust the expected roles accordingly
                if messages and messages[0]["role"] == "system":
                    # We need to ensure the first message after system is from user
                    expected_roles = ["user", "assistant"] * (len(context) // 2 + 1)
                
                for i, msg in enumerate(context):
                    # Get role and content
                    role = msg.get("role")
                    content = msg.get("content", "")
                    
                    # Skip messages with missing content
                    if not content:
                        continue
                        
                    # Ensure proper role alternation (user -> assistant -> user -> ...)
                    if i < len(expected_roles):
                        # Use the expected role for this position
                        valid_messages.append({"role": expected_roles[i], "content": content})
                    else:
                        # If we've gone beyond expected roles, just alternate from the last one
                        last_role = valid_messages[-1]["role"] if valid_messages else "assistant"
                        next_role = "user" if last_role == "assistant" else "assistant"
                        valid_messages.append({"role": next_role, "content": content})
                
                # Add the valid messages to our message list
                messages.extend(valid_messages)
                
            # Add the current prompt as a user message
            messages.append({"role": "user", "content": prompt})
            
            # Convert messages to the format expected by the model
            prompt_text = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Tokenize the input
            inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
            
            # Check if we're running on CPU or GPU
            is_cpu = self.model.device.type == "cpu"
            
            # Generate the response
            with torch.no_grad():
                # Adjust parameters based on device
                if is_cpu:
                    # CPU-optimized parameters
                    generation_config = {
                        "max_new_tokens": min(max_tokens, 512),
                        "temperature": temperature,
                        "do_sample": temperature > 0,
                        "pad_token_id": self.tokenizer.eos_token_id,
                        "use_cache": True,
                        "num_beams": 1,
                        "early_stopping": True
                    }
                else:
                    # GPU-optimized parameters
                    generation_config = {
                        "max_new_tokens": max_tokens,
                        "temperature": temperature,
                        "do_sample": temperature > 0,
                        "pad_token_id": self.tokenizer.eos_token_id,
                        "use_cache": True,
                        "top_p": 0.9,
                        "repetition_penalty": 1.1,
                        "num_beams": 1,
                        "early_stopping": True
                    }
                
                outputs = self.model.generate(inputs.input_ids, **generation_config)
            
            # Decode the output, skipping the input tokens
            generated_text = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], 
                skip_special_tokens=True
            )
            
            return generated_text
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
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
        # Build RAG context from documents
        if documents:
            rag_context = "The following information may be relevant to the question:\n\n"
            rag_context += "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(documents)])
            rag_context += "\n\nUsing the above information, answer the following question."
            
            # Combine with system prompt if provided
            enhanced_system_prompt = f"{system_prompt}\n\n{rag_context}" if system_prompt else rag_context
        else:
            enhanced_system_prompt = system_prompt
            
        # Use the regular generate function with enhanced context
        return await self.generate(
            prompt=prompt,
            system_prompt=enhanced_system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            context=context
        )