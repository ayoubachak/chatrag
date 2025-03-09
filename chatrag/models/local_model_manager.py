import logging
import os
import threading
from typing import Dict, Optional, Any
import torch

class ModelManager:
    """
    Singleton class to manage model instances and ensure they're only loaded once.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.logger = logging.getLogger(__name__)
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self._initialized = True
        self.logger.info("Model Manager initialized")
        
    def get_model_and_tokenizer(self, model_path: str):
        """
        Get a cached model and tokenizer or load them if not available.
        """
        if model_path in self.models and model_path in self.tokenizers:
            self.logger.debug(f"Using cached model: {model_path}")
            return self.models[model_path], self.tokenizers[model_path]
            
        self.logger.info(f"Loading model: {model_path}")
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            
            # Load tokenizer first
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                use_fast=True
            )
            
            # Determine device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Try different loading strategies with fallbacks
            model = None
            
            # First attempt: Try 4-bit quantization if on CUDA
            if device == "cuda":
                try:
                    self.logger.info("Attempting 4-bit quantization...")
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_enable_cpu_offload=True
                    )
                    
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16,
                        quantization_config=quantization_config,
                        device_map="auto",
                        low_cpu_mem_usage=True,
                        use_cache=True
                    )
                    self.logger.info("Model loaded with 4-bit quantization")
                except Exception as e:
                    self.logger.warning(f"4-bit quantization failed: {str(e)}")
                    model = None
                    
            # Second attempt: Try 8-bit quantization if 4-bit failed
            if model is None and device == "cuda":
                try:
                    self.logger.info("Attempting 8-bit quantization...")
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        load_in_8bit=True,
                        device_map="auto",
                        low_cpu_mem_usage=True,
                        use_cache=True
                    )
                    self.logger.info("Model loaded with 8-bit quantization")
                except Exception as e:
                    self.logger.warning(f"8-bit quantization failed: {str(e)}")
                    model = None
            
            # Third attempt: Try half precision without quantization
            if model is None and device == "cuda":
                try:
                    self.logger.info("Attempting half precision...")
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        low_cpu_mem_usage=True,
                        use_cache=True
                    )
                    self.logger.info("Model loaded in half precision")
                except Exception as e:
                    self.logger.warning(f"Half precision loading failed: {str(e)}")
                    model = None
            
            # Final fallback: CPU only if all GPU methods failed
            if model is None:
                self.logger.info("Loading model on CPU...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="cpu",
                    low_cpu_mem_usage=True,
                    use_cache=True
                )
                self.logger.info("Model loaded on CPU")
            
            # Cache model and tokenizer
            self.models[model_path] = model
            self.tokenizers[model_path] = tokenizer
            
            self.logger.info(f"Model {model_path} loaded successfully on {model.device}")
            return model, tokenizer
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_path}: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def clear_model(self, model_path: str):
        """
        Clear a specific model from cache to free memory.
        """
        if model_path in self.models:
            del self.models[model_path]
            self.logger.info(f"Model {model_path} removed from cache")
            
        if model_path in self.tokenizers:
            del self.tokenizers[model_path]
            self.logger.info(f"Tokenizer {model_path} removed from cache")
            
        # Force garbage collection
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def clear_all_models(self):
        """
        Clear all cached models to free memory.
        """
        self.models.clear()
        self.tokenizers.clear()
        
        # Force garbage collection
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.logger.info("All models cleared from cache")