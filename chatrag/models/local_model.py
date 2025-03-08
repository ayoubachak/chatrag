from typing import List, Dict, Any, Optional
import os

import torch
from .base import BaseLanguageModel

class LocalModel(BaseLanguageModel):
    """
    Implementation of the BaseLanguageModel for a local model using 
    the transformers library.
    """
    
    def __init__(self, model_path: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        """
        Initialize the local model.
        
        Args:
            model_path: Path to the local model or model identifier for HF
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        
    async def load_model(self):
        """
        Load the model and tokenizer if not already loaded.
        This is done as a separate method to avoid loading the model
        at initialization time, which can be slow.
        """
        if self.model is None or self.tokenizer is None:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch
                
                # Set up the device
                device = "cuda" if torch.cuda.is_available() else "cpu"
                
                # Load tokenizer and model
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    device_map=device
                )
                
                print(f"Model loaded successfully on {device}")
            except Exception as e:
                raise RuntimeError(f"Failed to load local model: {str(e)}")

    async def generate(self, 
                      prompt: str, 
                      system_prompt: Optional[str] = None,
                      temperature: float = 0.7,
                      max_tokens: int = 1024,
                      context: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Generate text using the local model.
        """
        await self.load_model()
        
        try:
            # Format the messages according to the model's expected format
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
                
            # Add previous context messages if provided
            if context:
                messages.extend(context)
                
            # Add the current prompt
            messages.append({"role": "user", "content": prompt})
            
            # Convert messages to the format expected by the model
            prompt_text = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Tokenize the input
            inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
            
            # Generate the response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode the output
            generated_text = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], 
                skip_special_tokens=True
            )
            
            return generated_text
        except Exception as e:
            return f"Error generating response with local model: {str(e)}"
            
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
        # Prepare RAG-enhanced prompt with retrieved documents
        rag_context = "The following information may be relevant to the question:\n\n"
        rag_context += "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(documents)])
        rag_context += "\n\nUsing the above information, answer the following question."
        
        # If system prompt is provided, combine it with RAG context
        if system_prompt:
            enhanced_system_prompt = f"{system_prompt}\n\n{rag_context}"
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