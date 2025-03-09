## .\app.py

```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from pathlib import Path

# Import routes
from routes import files, benchmark
from websockets import chat
from dotenv import load_dotenv

load_dotenv(override=True)


# Create FastAPI app
app = FastAPI(title="RAG Chat Application")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(files.router, prefix="/api", tags=["files"])
app.include_router(chat.router, tags=["chat"])
app.include_router(benchmark.router, prefix="/api", tags=["benchmark"])

# Create uploads directory if it doesn't exist
uploads_dir = Path("uploads")
uploads_dir.mkdir(exist_ok=True)

# Create vector store directory if it doesn't exist
vector_store_dir = Path("vector_store")
vector_store_dir.mkdir(exist_ok=True)

# Create directories for different vector store types
Path("chroma_store").mkdir(exist_ok=True)
Path("hybrid_store").mkdir(exist_ok=True)
Path("benchmark_results").mkdir(exist_ok=True)

@app.get("/api/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "healthy"}

# Mount static files (frontend build)
# This allows serving the frontend from the same server
try:
    static_files_dir = Path("../frontend/dist")
    if static_files_dir.exists():
        app.mount("/", StaticFiles(directory=str(static_files_dir), html=True), name="static")
except Exception as e:
    print(f"Warning: Could not mount static files: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 8000))
    
    # Run the server
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
```

## .\benchmark_rag.py

```python
#!/usr/bin/env python
import asyncio
import argparse
from rag.benchmark import RAGBenchmark

async def main():
    """
    Run the RAG benchmark with command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Benchmark different RAG implementations")
    
    parser.add_argument("--implementations", type=str, default="basic,faiss,chroma,hybrid",
                        help="Comma-separated list of implementations to benchmark (basic,faiss,chroma,hybrid)")
    parser.add_argument("--documents", type=int, default=1000,
                        help="Number of documents to use in the benchmark")
    parser.add_argument("--queries", type=int, default=100,
                        help="Number of queries to run")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Batch size for adding documents")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of results to retrieve per query")
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                        help="Directory to save benchmark results")
    
    args = parser.parse_args()
    
    # Parse implementations
    implementations = [impl.strip() for impl in args.implementations.split(",")]
    
    # Validate implementations
    valid_implementations = ["basic", "faiss", "chroma", "hybrid"]
    for impl in implementations:
        if impl not in valid_implementations:
            print(f"Warning: Invalid implementation '{impl}'. Valid options are: {', '.join(valid_implementations)}")
            implementations.remove(impl)
    
    if not implementations:
        print(f"No valid implementations specified. Using all: {', '.join(valid_implementations)}")
        implementations = valid_implementations
    
    # Run benchmark
    benchmark = RAGBenchmark(
        vector_store_types=implementations,
        benchmark_dir=args.output_dir
    )
    
    print(f"Running benchmark with {args.documents} documents and {args.queries} queries")
    print(f"Testing implementations: {', '.join(implementations)}")
    
    results = await benchmark.run_benchmark(
        num_documents=args.documents,
        num_queries=args.queries,
        batch_size=args.batch_size,
        top_k=args.top_k
    )
    
    benchmark.print_results(results)
    
    print(f"Benchmark results saved to {args.output_dir}")

if __name__ == "__main__":
    asyncio.run(main()) 
```

## .\config.py

```python
import os
from pydantic import BaseSettings
from typing import Optional, Dict, Any, List
from pathlib import Path
import dotenv

# Load environment variables from .env file if present
dotenv.load_dotenv()

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """
    # API settings
    API_PORT: int = 8000
    API_HOST: str = "0.0.0.0"
    
    # CORS settings
    CORS_ORIGINS: List[str] = ["*"]
    
    # File upload settings
    UPLOAD_DIR: str = "uploads"
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    
    # Vector store settings
    VECTOR_STORE_DIR: str = "vector_store"
    
    # HuggingFace settings
    HF_API_TOKEN: Optional[str] = os.environ.get("HF_API_TOKEN")
    HF_DEFAULT_MODEL: str = "mistralai/Mistral-7B-Instruct-v0.2"
    
    # LM Studio settings
    LM_STUDIO_API_BASE: str = "http://localhost:1234/v1"
    
    # Local model settings
    LOCAL_MODEL_PATH: str = "mistralai/Mistral-7B-Instruct-v0.2"
    
    # Embedding settings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # RAG settings
    RAG_NUM_RESULTS: int = 5
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Create global settings instance
settings = Settings()
```

## .\dependencies.py

```python

```

## .\run.py

```python
import uvicorn

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, ws="wsproto")
```

## .\__init__.py

```python
# backend package 
```

## .\models\base.py

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseLanguageModel(ABC):
    """
    Abstract base class for all language model implementations.
    This ensures a consistent interface regardless of the model source.
    """
    
    @abstractmethod
    async def generate(self, 
                      prompt: str, 
                      system_prompt: Optional[str] = None,
                      temperature: float = 0.7,
                      max_tokens: int = 1024,
                      context: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Generate a response based on the prompt and optional context.
        
        Args:
            prompt: The user's message to respond to
            system_prompt: Optional system prompt to guide the model
            temperature: Controls randomness (higher = more random)
            max_tokens: Maximum number of tokens to generate
            context: Optional list of previous messages for context
            
        Returns:
            The generated text response
        """
        pass
    
    @abstractmethod
    async def generate_with_rag(self,
                              prompt: str,
                              documents: List[str],
                              system_prompt: Optional[str] = None,
                              temperature: float = 0.7,
                              max_tokens: int = 1024,
                              context: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Generate a response with RAG-enhanced context.
        
        Args:
            prompt: The user's message to respond to
            documents: List of retrieved document snippets to include as context
            system_prompt: Optional system prompt to guide the model
            temperature: Controls randomness (higher = more random)
            max_tokens: Maximum number of tokens to generate
            context: Optional list of previous messages for context
            
        Returns:
            The generated text response
        """
        pass
```

## .\models\huggingface.py

```python
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
```

## .\models\lm_studio.py

```python
import os
import httpx
from typing import List, Dict, Any, Optional
from .base import BaseLanguageModel
from openai import OpenAI

class LMStudioModel(BaseLanguageModel):
    """
    Implementation of the BaseLanguageModel for LM Studio's local API.
    This uses the OpenAI-compatible API that LM Studio provides.
    """
    
    def __init__(self, api_base: str = "http://localhost:1234/v1"):
        """
        Initialize the LM Studio model client.
        
        Args:
            api_base: Base URL for the LM Studio API
        """
        self.api_base = api_base
        # Replace httpx client with OpenAI client
        self.client = OpenAI(base_url=self.api_base, api_key="not-needed")

    async def generate(self, 
                      prompt: str, 
                      system_prompt: Optional[str] = None,
                      temperature: float = 0.7,
                      max_tokens: int = 1024,
                      context: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Generate text using the LM Studio API.
        """
        # Format the messages according to OpenAI's chat format
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        # Add previous context messages if provided, ensuring they follow the alternating pattern
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
                # Also check if role is missing or undefined
                if "role" not in msg or msg.get("role") is None:
                    valid_context = False
                    break
            
            if valid_context:
                messages.extend(context)
            else:
                # If context doesn't have proper alternating pattern, reconstruct it
                current_role = "user"
                for msg in context:
                    if "content" in msg and msg.get("content") is not None:
                        messages.append({"role": current_role, "content": msg.get("content", "")})
                        # Toggle between user and assistant
                        current_role = "assistant" if current_role == "user" else "user"
            
        # Add the current prompt
        messages.append({"role": "user", "content": prompt})
        
        try:
            # Use async version of the OpenAI client if available, otherwise use synchronous version
            try:
                # For newer versions of the OpenAI client that support async
                chat_completion = await self.client.chat.completions.create(
                    messages=messages,
                    model="local-model", # LM Studio doesn't need a specific model name
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            except (TypeError, AttributeError):
                # Fallback for older versions or if async is not supported
                chat_completion = self.client.chat.completions.create(
                    messages=messages,
                    model="local-model", # LM Studio doesn't need a specific model name
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Error generating response with LM Studio: {str(e)}"
            
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
```

## .\models\local_model.py

```python
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
```

## .\models\local_model_manager.py

```python
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
```

## .\models\__init__.py

```python

```

## .\rag\benchmark.py

```python
import time
import asyncio
import random
import string
import numpy as np
from typing import List, Dict, Any, Literal
from pathlib import Path
import os
import json

from .pipeline import RAGPipeline
from .logger import rag_logger

class RAGBenchmark:
    """
    Benchmark utility to compare the performance of different RAG implementations.
    """
    
    def __init__(self, 
                 vector_store_types: List[Literal["basic", "faiss", "chroma", "hybrid"]] = ["basic", "faiss", "chroma", "hybrid"],
                 benchmark_dir: str = "benchmark_results"):
        """
        Initialize the benchmark utility.
        
        Args:
            vector_store_types: List of vector store types to benchmark
            benchmark_dir: Directory to save benchmark results
        """
        self.vector_store_types = vector_store_types
        self.benchmark_dir = Path(benchmark_dir)
        self.benchmark_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize pipelines
        self.pipelines = {}
        for store_type in vector_store_types:
            self.pipelines[store_type] = RAGPipeline(vector_store_type=store_type)
            
        rag_logger.info(f"Initialized RAG benchmark with store types: {', '.join(vector_store_types)}")
        
    async def generate_test_data(self, 
                               num_documents: int = 100, 
                               embedding_dim: int = 384,
                               doc_length: int = 200):
        """
        Generate synthetic test data for benchmarking.
        
        Args:
            num_documents: Number of test documents to generate
            embedding_dim: Dimension of embeddings
            doc_length: Average length of documents in words
            
        Returns:
            Tuple of (documents, embeddings, queries)
        """
        rag_logger.info(f"Generating {num_documents} test documents with embedding dimension {embedding_dim}")
        
        # Generate random documents
        documents = []
        for i in range(num_documents):
            # Generate random text
            words = []
            for _ in range(random.randint(doc_length - 50, doc_length + 50)):
                word_length = random.randint(3, 10)
                word = ''.join(random.choice(string.ascii_lowercase) for _ in range(word_length))
                words.append(word)
                
            content = ' '.join(words)
            
            # Create document
            documents.append({
                "content": content,
                "metadata": {
                    "id": f"doc_{i}",
                    "source": "synthetic",
                    "length": len(content)
                }
            })
            
        # Generate random embeddings (normalized)
        embeddings = []
        for _ in range(num_documents):
            embedding = np.random.randn(embedding_dim).astype(np.float32)
            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding.tolist())
            
        # Generate random query embeddings
        queries = []
        for i in range(10):  # 10 test queries
            query_embedding = np.random.randn(embedding_dim).astype(np.float32)
            # Normalize
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            queries.append({
                "id": f"query_{i}",
                "embedding": query_embedding.tolist()
            })
            
        rag_logger.info(f"Generated {len(documents)} documents, {len(embeddings)} embeddings, and {len(queries)} queries")
        return documents, embeddings, queries
        
    async def run_benchmark(self, 
                          num_documents: int = 1000, 
                          num_queries: int = 100,
                          batch_size: int = 100,
                          top_k: int = 5):
        """
        Run a benchmark comparing different RAG implementations.
        
        Args:
            num_documents: Total number of documents to use
            num_queries: Number of queries to run
            batch_size: Batch size for adding documents
            top_k: Number of results to retrieve per query
            
        Returns:
            Benchmark results
        """
        rag_logger.info(f"Starting RAG benchmark with {num_documents} documents and {num_queries} queries")
        
        # Generate test data
        documents, embeddings, queries = await self.generate_test_data(num_documents=num_documents)
        
        # Results dictionary
        results = {
            "parameters": {
                "num_documents": num_documents,
                "num_queries": num_queries,
                "batch_size": batch_size,
                "top_k": top_k
            },
            "results": {}
        }
        
        # Test each pipeline
        for store_type, pipeline in self.pipelines.items():
            rag_logger.info(f"Benchmarking {store_type} RAG implementation")
            
            # Measure document addition time
            add_times = []
            for i in range(0, num_documents, batch_size):
                end_idx = min(i + batch_size, num_documents)
                batch_docs = documents[i:end_idx]
                batch_embeddings = embeddings[i:end_idx]
                
                start_time = time.time()
                await pipeline.vector_store.add_documents(batch_docs, batch_embeddings)
                end_time = time.time()
                
                add_times.append(end_time - start_time)
                rag_logger.info(f"Added batch {i//batch_size + 1} ({len(batch_docs)} documents) to {store_type} in {add_times[-1]:.4f} seconds")
            
            # Measure query time
            query_times = []
            for i in range(min(num_queries, len(queries))):
                query = queries[i % len(queries)]
                
                start_time = time.time()
                results_list = await pipeline.vector_store.search(query["embedding"], top_k=top_k)
                end_time = time.time()
                
                query_times.append(end_time - start_time)
                
                if i % 10 == 0:
                    rag_logger.info(f"Ran query {i+1}/{num_queries} on {store_type} in {query_times[-1]:.4f} seconds")
            
            # Measure save time
            start_time = time.time()
            save_path = await pipeline.save(f"benchmark_{store_type}.pkl")
            end_time = time.time()
            save_time = end_time - start_time
            
            rag_logger.info(f"Saved {store_type} RAG pipeline in {save_time:.4f} seconds")
            
            # Measure load time
            start_time = time.time()
            loaded_pipeline = await RAGPipeline.load(save_path, vector_store_type=store_type)
            end_time = time.time()
            load_time = end_time - start_time
            
            rag_logger.info(f"Loaded {store_type} RAG pipeline in {load_time:.4f} seconds")
            
            # Store results
            results["results"][store_type] = {
                "add_time": {
                    "total": sum(add_times),
                    "average_per_batch": sum(add_times) / len(add_times),
                    "average_per_document": sum(add_times) / num_documents
                },
                "query_time": {
                    "total": sum(query_times),
                    "average": sum(query_times) / len(query_times)
                },
                "save_time": save_time,
                "load_time": load_time
            }
            
        # Save benchmark results
        timestamp = int(time.time())
        result_path = self.benchmark_dir / f"benchmark_results_{timestamp}.json"
        with open(result_path, "w") as f:
            json.dump(results, f, indent=2)
            
        rag_logger.info(f"Benchmark complete. Results saved to {result_path}")
        return results
        
    def print_results(self, results: Dict[str, Any]):
        """
        Print benchmark results in a readable format.
        
        Args:
            results: Benchmark results
        """
        print("\n===== RAG BENCHMARK RESULTS =====\n")
        print(f"Documents: {results['parameters']['num_documents']}")
        print(f"Queries: {results['parameters']['num_queries']}")
        print(f"Batch size: {results['parameters']['batch_size']}")
        print(f"Top-k: {results['parameters']['top_k']}")
        print("\n")
        
        # Print table header
        header = f"{'Implementation':<10} | {'Add Time (s)':<12} | {'Query Time (ms)':<15} | {'Save Time (s)':<12} | {'Load Time (s)':<12}"
        print(header)
        print("-" * len(header))
        
        # Print results for each implementation
        for store_type, metrics in results["results"].items():
            add_time = metrics["add_time"]["total"]
            query_time = metrics["query_time"]["average"] * 1000  # Convert to ms
            save_time = metrics["save_time"]
            load_time = metrics["load_time"]
            
            print(f"{store_type:<10} | {add_time:<12.4f} | {query_time:<15.4f} | {save_time:<12.4f} | {load_time:<12.4f}")
            
        print("\n")
        
        # Print summary
        print("Summary:")
        fastest_add = min(results["results"].items(), key=lambda x: x[1]["add_time"]["total"])[0]
        fastest_query = min(results["results"].items(), key=lambda x: x[1]["query_time"]["average"])[0]
        fastest_save = min(results["results"].items(), key=lambda x: x[1]["save_time"])[0]
        fastest_load = min(results["results"].items(), key=lambda x: x[1]["load_time"])[0]
        
        print(f"- Fastest document addition: {fastest_add}")
        print(f"- Fastest query: {fastest_query}")
        print(f"- Fastest save: {fastest_save}")
        print(f"- Fastest load: {fastest_load}")
        
        # Overall recommendation
        print("\nRecommendation:")
        if fastest_query == "faiss":
            print("- For speed-critical applications with many queries: Use FAISS")
        elif fastest_query == "hybrid":
            print("- For balanced performance and persistence: Use Hybrid")
        elif fastest_query == "chroma":
            print("- For persistence and advanced filtering: Use ChromaDB")
        else:
            print("- For simple use cases: Use Basic")
            
        print("\n===================================\n")


async def run_benchmark():
    """
    Run a RAG benchmark from the command line.
    """
    benchmark = RAGBenchmark()
    results = await benchmark.run_benchmark()
    benchmark.print_results(results)
    
    
if __name__ == "__main__":
    asyncio.run(run_benchmark()) 
```

## .\rag\chroma_retriever.py

```python
from typing import List, Dict, Any, Optional, Union
import os
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
import uuid
import json
from .logger import retriever_logger

class ChromaVectorStore:
    """
    A ChromaDB-based vector store for persistent document retrieval.
    
    This implementation uses ChromaDB for efficient vector storage and retrieval,
    with persistence and collection management capabilities.
    """
    
    def __init__(self, storage_dir: str = "chroma_store"):
        """
        Initialize the ChromaDB vector store.
        
        Args:
            storage_dir: Directory to save the ChromaDB data
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client with persistent storage
        self.client = chromadb.PersistentClient(path=str(self.storage_dir))
        
        # Unique ID for this session
        self.session_id = str(uuid.uuid4())
        self.collection_name = f"documents_{self.session_id}"
        
        # Create a collection for this session
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"session_id": self.session_id}
        )
        
        retriever_logger.info(f"Initialized ChromaDB vector store with session ID: {self.session_id}")
        
    async def add_documents(self, 
                           documents: List[Dict[str, Any]], 
                           embeddings: List[List[float]]):
        """
        Add documents and their embeddings to the ChromaDB vector store.
        
        Args:
            documents: List of document dictionaries (content and metadata)
            embeddings: List of embedding vectors for the documents
        """
        retriever_logger.info(f"Adding {len(documents)} documents to ChromaDB vector store")
        
        if len(documents) != len(embeddings):
            retriever_logger.warning(f"Number of documents ({len(documents)}) and embeddings ({len(embeddings)}) do not match")
            raise ValueError("Number of documents and embeddings must match")
        
        # Prepare data for ChromaDB
        ids = [f"{self.session_id}_{i}" for i in range(len(documents))]
        document_texts = [doc["content"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        
        # Add documents to ChromaDB collection
        try:
            self.collection.add(
                ids=ids,
                documents=document_texts,
                embeddings=embeddings,
                metadatas=metadatas
            )
            retriever_logger.info(f"Successfully added {len(documents)} documents to ChromaDB collection")
        except Exception as e:
            retriever_logger.error(f"Error adding documents to ChromaDB: {str(e)}", exc_info=True)
            
    async def search(self, 
                    query_embedding: List[float], 
                    top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query embedding using ChromaDB.
        
        Args:
            query_embedding: Embedding vector for the query
            top_k: Number of top results to return
            
        Returns:
            List of document dictionaries with similarity scores
        """
        retriever_logger.info(f"Searching ChromaDB vector store with query embedding")
        
        try:
            # Query the collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Process results
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]
            ids = results.get("ids", [[]])[0]
            
            # Convert distances to similarity scores (ChromaDB returns distances, not similarities)
            # For cosine distance, similarity = 1 - distance
            scores = [1 - dist for dist in distances]
            
            # Prepare results
            result_docs = []
            for i in range(len(documents)):
                result_docs.append({
                    "id": ids[i],
                    "content": documents[i],
                    "metadata": metadatas[i],
                    "score": float(scores[i])
                })
                
            retriever_logger.info(f"Returning {len(result_docs)} search results from ChromaDB")
            for i, result in enumerate(result_docs):
                retriever_logger.debug(f"Result {i+1}: Score {result['score']:.4f}, Content: '{result['content'][:50]}...'")
                
            return result_docs
        except Exception as e:
            retriever_logger.error(f"Error searching ChromaDB: {str(e)}", exc_info=True)
            return []
            
    async def save(self, filepath: Optional[str] = None):
        """
        Save the ChromaDB vector store metadata to disk.
        
        Note: ChromaDB already persists data, this just saves the session info.
        
        Args:
            filepath: Optional path to save the metadata
            
        Returns:
            Path to the saved metadata
        """
        if filepath is None:
            filepath = str(self.storage_dir / f"chroma_metadata_{self.session_id}.json")
            
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Prepare metadata
        metadata = {
            "session_id": self.session_id,
            "collection_name": self.collection_name,
            "storage_dir": str(self.storage_dir)
        }
        
        retriever_logger.info(f"Saving ChromaDB metadata to {filepath}")
        
        # Save metadata to disk
        try:
            with open(filepath, "w") as f:
                json.dump(metadata, f)
            retriever_logger.info(f"Successfully saved ChromaDB metadata to {filepath}")
        except Exception as e:
            retriever_logger.error(f"Error saving ChromaDB metadata to {filepath}: {str(e)}", exc_info=True)
            
        return filepath
        
    @classmethod
    async def load(cls, filepath: str):
        """
        Load a ChromaDB vector store from metadata.
        
        Args:
            filepath: Path to the saved metadata
            
        Returns:
            A new ChromaVectorStore instance connected to the existing collection
        """
        retriever_logger.info(f"Loading ChromaDB vector store from: {filepath}")
        
        try:
            # Check if the file exists
            if not os.path.exists(filepath):
                retriever_logger.warning(f"ChromaDB metadata file does not exist: {filepath}")
                return cls()
                
            # Load metadata
            with open(filepath, "r") as f:
                metadata = json.load(f)
                
            # Extract necessary information
            session_id = metadata.get("session_id")
            collection_name = metadata.get("collection_name")
            storage_dir = metadata.get("storage_dir")
            
            if not all([session_id, collection_name, storage_dir]):
                retriever_logger.warning("Missing required metadata for ChromaDB")
                return cls()
                
            # Create a new instance with the same storage directory
            store = cls(storage_dir=storage_dir)
            
            # Override the auto-generated session and collection
            store.session_id = session_id
            
            # Get the existing collection instead of creating a new one
            try:
                # Delete the auto-created collection
                store.client.delete_collection(store.collection_name)
                
                # Connect to the existing collection
                store.collection_name = collection_name
                store.collection = store.client.get_collection(name=collection_name)
                
                # Get collection info
                collection_count = store.collection.count()
                retriever_logger.info(f"Successfully loaded ChromaDB collection with {collection_count} documents")
            except Exception as e:
                retriever_logger.error(f"Error connecting to existing ChromaDB collection: {str(e)}", exc_info=True)
                # The auto-created collection will be used as fallback
            
            return store
        except Exception as e:
            retriever_logger.error(f"Error loading ChromaDB from {filepath}: {str(e)}", exc_info=True)
            # Return a new store
            return cls() 
```

## .\rag\document_loader.py

```python
import os
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import tempfile
import uuid

class DocumentLoader:
    """
    Utility class for loading and processing documents from various file formats.
    """
    
    def __init__(self, upload_dir: str = "uploads"):
        """
        Initialize the document loader.
        
        Args:
            upload_dir: Directory to store uploaded files
        """
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
    async def save_uploaded_file(self, file) -> str:
        """
        Save an uploaded file to disk and return its path.
        
        Args:
            file: The uploaded file object from FastAPI
            
        Returns:
            Path to the saved file
        """
        # Create a unique filename to avoid collisions
        filename = f"{uuid.uuid4()}_{file.filename}"
        file_path = self.upload_dir / filename
        
        # Write the file to disk
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
            
        return str(file_path)
    
    async def load_document(self, file_path: str) -> List[Dict[str, str]]:
        """
        Load a document and split it into chunks.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of document chunks with metadata
        """
        file_ext = Path(file_path).suffix.lower()
        
        try:
            if file_ext == ".txt":
                return await self._load_text(file_path)
            elif file_ext == ".pdf":
                return await self._load_pdf(file_path)
            elif file_ext in [".docx", ".doc"]:
                return await self._load_docx(file_path)
            elif file_ext in [".csv", ".xlsx", ".xls"]:
                return await self._load_tabular(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
        except Exception as e:
            print(f"Error loading document {file_path}: {str(e)}")
            return []
            
    async def _load_text(self, file_path: str) -> List[Dict[str, str]]:
        """
        Load and chunk a text file.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            
        # Simple chunking by paragraphs
        chunks = [p.strip() for p in text.split("\n\n") if p.strip()]
        
        return [
            {
                "content": chunk,
                "metadata": {
                    "source": Path(file_path).name,
                    "chunk_id": i
                }
            }
            for i, chunk in enumerate(chunks)
        ]
        
    async def _load_pdf(self, file_path: str) -> List[Dict[str, str]]:
        """
        Load and chunk a PDF file.
        """
        try:
            from pypdf import PdfReader
            
            reader = PdfReader(file_path)
            text_chunks = []
            
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():
                    text_chunks.append({
                        "content": text.strip(),
                        "metadata": {
                            "source": Path(file_path).name,
                            "page": i + 1,
                            "chunk_id": i
                        }
                    })
                    
            return text_chunks
        except ImportError:
            raise ImportError("pypdf is required for PDF processing. Install with: pip install pypdf")
            
    async def _load_docx(self, file_path: str) -> List[Dict[str, str]]:
        """
        Load and chunk a Word document.
        """
        try:
            import docx
            
            doc = docx.Document(file_path)
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            
            return [
                {
                    "content": para,
                    "metadata": {
                        "source": Path(file_path).name,
                        "chunk_id": i
                    }
                }
                for i, para in enumerate(paragraphs)
            ]
        except ImportError:
            raise ImportError("python-docx is required for DOCX processing. Install with: pip install python-docx")
            
    async def _load_tabular(self, file_path: str) -> List[Dict[str, str]]:
        """
        Load and process tabular data (CSV, Excel).
        """
        file_ext = Path(file_path).suffix.lower()
        
        try:
            import pandas as pd
            
            if file_ext == ".csv":
                df = pd.read_csv(file_path)
            else:  # Excel files
                df = pd.read_excel(file_path)
                
            # Convert each row to a text chunk
            chunks = []
            for i, row in enumerate(df.to_dict("records")):
                # Format the row as a string
                content = "\n".join([f"{k}: {v}" for k, v in row.items()])
                chunks.append({
                    "content": content,
                    "metadata": {
                        "source": Path(file_path).name,
                        "row": i + 1,
                        "chunk_id": i
                    }
                })
                
            return chunks
        except ImportError:
            raise ImportError("pandas is required for tabular data processing. Install with: pip install pandas openpyxl")
```

## .\rag\embedding.py

```python
from typing import List, Dict, Any, Optional, Union
import numpy as np
import os
from .logger import embedding_logger

class EmbeddingGenerator:
    """
    Generate embeddings for text using various models.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the embedding model to use
        """
        self.model_name = model_name
        self.model = None
        
    async def load_model(self):
        """
        Load the embedding model if not already loaded.
        """
        if self.model is None:
            embedding_logger.info(f"Loading embedding model: {self.model_name}")
            try:
                from sentence_transformers import SentenceTransformer
                
                # Load the model
                self.model = SentenceTransformer(self.model_name)
                embedding_logger.info(f"Embedding model loaded successfully: {self.model_name}")
                
                # Test the model with a simple input
                test_embedding = self.model.encode(["Test sentence"])
                embedding_logger.info(f"Model test successful. Embedding shape: {test_embedding.shape}")
                
            except ImportError as e:
                embedding_logger.error(f"Error importing sentence-transformers: {str(e)}")
                raise ImportError("sentence-transformers is required for embeddings. Install with: pip install -U sentence-transformers")
            except Exception as e:
                embedding_logger.error(f"Error loading embedding model: {str(e)}", exc_info=True)
                raise RuntimeError(f"Failed to load embedding model: {str(e)}")
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            embedding_logger.warning("No texts provided for embedding generation")
            return []
            
        await self.load_model()
        
        embedding_logger.info(f"Generating embeddings for {len(texts)} texts")
        
        # Generate embeddings
        try:
            embeddings = self.model.encode(texts)
            embedding_logger.info(f"Successfully generated embeddings with shape: {embeddings.shape if hasattr(embeddings, 'shape') else len(embeddings)}")
            
            # Convert numpy arrays to lists for JSON serialization
            embeddings_list = embeddings.tolist()
            
            # Verify embeddings
            if len(embeddings_list) != len(texts):
                embedding_logger.warning(f"Number of generated embeddings ({len(embeddings_list)}) does not match number of texts ({len(texts)})")
                
            # Check for NaN or zero embeddings
            for i, emb in enumerate(embeddings_list):
                if np.isnan(np.sum(emb)) or np.allclose(emb, 0):
                    embedding_logger.warning(f"Embedding {i} contains NaN or all zeros")
                    
            return embeddings_list
        except Exception as e:
            embedding_logger.error(f"Error generating embeddings: {str(e)}", exc_info=True)
            # Return empty embeddings as fallback
            return [[0.0] * 384] * len(texts)  # 384 is the dimension for all-MiniLM-L6-v2
        
    async def generate_single_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for a single text.
        
        Args:
            text: Text string to embed
            
        Returns:
            Embedding vector
        """
        embedding_logger.info(f"Generating single embedding for text: '{text[:50]}...'")
        embeddings = await self.generate_embeddings([text])
        return embeddings[0]
```

## .\rag\faiss_retriever.py

```python
from typing import List, Dict, Any, Optional, Union
import numpy as np
import os
import pickle
import faiss
from pathlib import Path
import uuid
from .logger import retriever_logger

class FAISSVectorStore:
    """
    A FAISS-based vector store for efficient document retrieval.
    
    This implementation uses FAISS for fast similarity search, which is
    significantly more efficient than the basic vector store for large collections.
    """
    
    def __init__(self, storage_dir: str = "vector_store"):
        """
        Initialize the FAISS vector store.
        
        Args:
            storage_dir: Directory to save the vector store
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory storage for document data
        self.document_ids = []
        self.document_texts = []
        self.document_metadata = []
        
        # FAISS index will be initialized when first embeddings are added
        self.index = None
        self.embedding_dim = None
        
        # Unique ID for this session
        self.session_id = str(uuid.uuid4())
        retriever_logger.info(f"Initialized FAISS vector store with session ID: {self.session_id}")
        
    async def add_documents(self, 
                           documents: List[Dict[str, Any]], 
                           embeddings: List[List[float]]):
        """
        Add documents and their embeddings to the FAISS vector store.
        
        Args:
            documents: List of document dictionaries (content and metadata)
            embeddings: List of embedding vectors for the documents
        """
        retriever_logger.info(f"Adding {len(documents)} documents to FAISS vector store")
        
        if len(documents) != len(embeddings):
            retriever_logger.warning(f"Number of documents ({len(documents)}) and embeddings ({len(embeddings)}) do not match")
            raise ValueError("Number of documents and embeddings must match")
            
        # Convert embeddings to numpy array
        embeddings_np = np.array(embeddings, dtype=np.float32)
        
        # Initialize FAISS index if this is the first batch
        if self.index is None:
            self.embedding_dim = embeddings_np.shape[1]
            retriever_logger.info(f"Initializing FAISS index with dimension {self.embedding_dim}")
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity with normalized vectors
        
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(embeddings_np)
        
        # Add embeddings to FAISS index
        self.index.add(embeddings_np)
        
        # Store document data
        start_idx = len(self.document_ids)
        for i, doc in enumerate(documents):
            doc_id = f"{self.session_id}_{start_idx + i}"
            self.document_ids.append(doc_id)
            self.document_texts.append(doc["content"])
            self.document_metadata.append(doc["metadata"])
            
        retriever_logger.info(f"FAISS vector store now contains {len(self.document_ids)} documents")
            
    async def search(self, 
                    query_embedding: List[float], 
                    top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query embedding using FAISS.
        
        Args:
            query_embedding: Embedding vector for the query
            top_k: Number of top results to return
            
        Returns:
            List of document dictionaries with similarity scores
        """
        retriever_logger.info(f"Searching FAISS vector store with {len(self.document_ids)} documents")
        
        if self.index is None or self.index.ntotal == 0:
            retriever_logger.warning("No embeddings in FAISS vector store")
            return []
            
        # Convert query to numpy array and normalize
        query_np = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_np)
        
        # Search the FAISS index
        scores, indices = self.index.search(query_np, min(top_k, len(self.document_ids)))
        
        # Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.document_ids):
                continue  # Skip invalid indices
                
            results.append({
                "id": self.document_ids[idx],
                "content": self.document_texts[idx],
                "metadata": self.document_metadata[idx],
                "score": float(scores[0][i])
            })
            
        retriever_logger.info(f"Returning {len(results)} search results from FAISS")
        for i, result in enumerate(results):
            retriever_logger.debug(f"Result {i+1}: Score {result['score']:.4f}, Content: '{result['content'][:50]}...'")
            
        return results
        
    async def save(self, filepath: Optional[str] = None):
        """
        Save the FAISS vector store to disk.
        
        Args:
            filepath: Optional path to save the vector store
            
        Returns:
            Path to the saved vector store
        """
        if filepath is None:
            filepath = str(self.storage_dir / f"faiss_vector_store_{self.session_id}.pkl")
            
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save FAISS index to a separate file
        index_path = f"{filepath}.index"
        
        retriever_logger.info(f"Saving FAISS index to {index_path}")
        if self.index is not None:
            faiss.write_index(self.index, index_path)
        
        # Prepare metadata to save
        data = {
            "document_ids": self.document_ids,
            "document_texts": self.document_texts,
            "document_metadata": self.document_metadata,
            "embedding_dim": self.embedding_dim,
            "session_id": self.session_id
        }
        
        retriever_logger.info(f"Saving FAISS vector store metadata with {len(self.document_ids)} documents to {filepath}")
        
        # Save metadata to disk
        try:
            with open(filepath, "wb") as f:
                pickle.dump(data, f)
            retriever_logger.info(f"Successfully saved FAISS vector store to {filepath}")
        except Exception as e:
            retriever_logger.error(f"Error saving FAISS vector store to {filepath}: {str(e)}", exc_info=True)
            
        return filepath
        
    @classmethod
    async def load(cls, filepath: str):
        """
        Load a FAISS vector store from disk.
        
        Args:
            filepath: Path to the saved vector store
            
        Returns:
            A new FAISSVectorStore instance
        """
        retriever_logger.info(f"Loading FAISS vector store from: {filepath}")
        
        try:
            # Check if the file exists
            if not os.path.exists(filepath):
                retriever_logger.warning(f"FAISS vector store file does not exist: {filepath}")
                return cls()
                
            # Load metadata
            with open(filepath, "rb") as f:
                data = pickle.load(f)
                
            # Create a new instance
            store = cls()
            
            # Restore metadata
            store.document_ids = data["document_ids"]
            store.document_texts = data["document_texts"]
            store.document_metadata = data["document_metadata"]
            store.embedding_dim = data["embedding_dim"]
            store.session_id = data["session_id"]
            
            # Load FAISS index if it exists
            index_path = f"{filepath}.index"
            if os.path.exists(index_path) and store.embedding_dim is not None:
                store.index = faiss.read_index(index_path)
                retriever_logger.info(f"Loaded FAISS index with {store.index.ntotal} vectors")
            else:
                retriever_logger.warning(f"FAISS index file not found: {index_path}")
                # Initialize an empty index if dimension is known
                if store.embedding_dim is not None:
                    store.index = faiss.IndexFlatIP(store.embedding_dim)
            
            retriever_logger.info(f"Successfully loaded FAISS vector store with {len(store.document_ids)} documents")
            return store
        except Exception as e:
            retriever_logger.error(f"Error loading FAISS vector store from {filepath}: {str(e)}", exc_info=True)
            # Return an empty store
            return cls() 
```

## .\rag\hybrid_retriever.py

```python
from typing import List, Dict, Any, Optional, Union
import numpy as np
import os
import pickle
import faiss
import chromadb
from pathlib import Path
import uuid
import json
import shutil
from .logger import retriever_logger

class HybridVectorStore:
    """
    A hybrid vector store that combines FAISS for fast retrieval and ChromaDB for persistence.
    
    This implementation uses FAISS for efficient similarity search in memory,
    while leveraging ChromaDB for persistent storage and collection management.
    """
    
    def __init__(self, storage_dir: str = "hybrid_store"):
        """
        Initialize the hybrid vector store.
        
        Args:
            storage_dir: Directory to save the vector store data
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client with persistent storage
        self.chroma_dir = self.storage_dir / "chroma"
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.chroma_dir))
        
        # Unique ID for this session
        self.session_id = str(uuid.uuid4())
        self.collection_name = f"documents_{self.session_id}"
        
        # Create a collection for this session
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"session_id": self.session_id}
        )
        
        # FAISS index for fast in-memory search
        self.index = None
        self.embedding_dim = None
        
        # Document tracking
        self.document_ids = []
        self.document_texts = []
        self.document_metadata = []
        
        retriever_logger.info(f"Initialized hybrid vector store with session ID: {self.session_id}")
        
    async def add_documents(self, 
                           documents: List[Dict[str, Any]], 
                           embeddings: List[List[float]]):
        """
        Add documents and their embeddings to both FAISS and ChromaDB.
        
        Args:
            documents: List of document dictionaries (content and metadata)
            embeddings: List of embedding vectors for the documents
        """
        retriever_logger.info(f"Adding {len(documents)} documents to hybrid vector store")
        
        if len(documents) != len(embeddings):
            retriever_logger.warning(f"Number of documents ({len(documents)}) and embeddings ({len(embeddings)}) do not match")
            raise ValueError("Number of documents and embeddings must match")
        
        # Prepare document IDs
        start_idx = len(self.document_ids)
        ids = [f"{self.session_id}_{start_idx + i}" for i in range(len(documents))]
        
        # Add to document tracking
        for i, doc in enumerate(documents):
            self.document_ids.append(ids[i])
            self.document_texts.append(doc["content"])
            self.document_metadata.append(doc["metadata"])
        
        # Add to ChromaDB for persistence
        try:
            document_texts = [doc["content"] for doc in documents]
            metadatas = [doc["metadata"] for doc in documents]
            
            self.collection.add(
                ids=ids,
                documents=document_texts,
                embeddings=embeddings,
                metadatas=metadatas
            )
            retriever_logger.info(f"Added {len(documents)} documents to ChromaDB collection")
        except Exception as e:
            retriever_logger.error(f"Error adding documents to ChromaDB: {str(e)}", exc_info=True)
        
        # Add to FAISS for fast retrieval
        try:
            # Convert embeddings to numpy array
            embeddings_np = np.array(embeddings, dtype=np.float32)
            
            # Initialize FAISS index if this is the first batch
            if self.index is None:
                self.embedding_dim = embeddings_np.shape[1]
                retriever_logger.info(f"Initializing FAISS index with dimension {self.embedding_dim}")
                self.index = faiss.IndexFlatIP(self.embedding_dim)
            
            # Normalize vectors for cosine similarity
            faiss.normalize_L2(embeddings_np)
            
            # Add embeddings to FAISS index
            self.index.add(embeddings_np)
            retriever_logger.info(f"Added {len(documents)} documents to FAISS index")
        except Exception as e:
            retriever_logger.error(f"Error adding documents to FAISS: {str(e)}", exc_info=True)
            
    async def search(self, 
                    query_embedding: List[float], 
                    top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query embedding using FAISS for speed.
        Falls back to ChromaDB if FAISS search fails.
        
        Args:
            query_embedding: Embedding vector for the query
            top_k: Number of top results to return
            
        Returns:
            List of document dictionaries with similarity scores
        """
        retriever_logger.info(f"Searching hybrid vector store with query embedding")
        
        # Try FAISS first for speed
        if self.index is not None and self.index.ntotal > 0:
            try:
                # Convert query to numpy array and normalize
                query_np = np.array([query_embedding], dtype=np.float32)
                faiss.normalize_L2(query_np)
                
                # Search the FAISS index
                scores, indices = self.index.search(query_np, min(top_k, len(self.document_ids)))
                
                # Prepare results
                results = []
                for i, idx in enumerate(indices[0]):
                    if idx < 0 or idx >= len(self.document_ids):
                        continue  # Skip invalid indices
                        
                    results.append({
                        "id": self.document_ids[idx],
                        "content": self.document_texts[idx],
                        "metadata": self.document_metadata[idx],
                        "score": float(scores[0][i])
                    })
                    
                retriever_logger.info(f"Returning {len(results)} search results from FAISS")
                return results
            except Exception as e:
                retriever_logger.error(f"Error searching FAISS, falling back to ChromaDB: {str(e)}", exc_info=True)
        
        # Fall back to ChromaDB if FAISS fails or is empty
        try:
            # Query the collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Process results
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]
            ids = results.get("ids", [[]])[0]
            
            # Convert distances to similarity scores (ChromaDB returns distances, not similarities)
            # For cosine distance, similarity = 1 - distance
            scores = [1 - dist for dist in distances]
            
            # Prepare results
            result_docs = []
            for i in range(len(documents)):
                result_docs.append({
                    "id": ids[i],
                    "content": documents[i],
                    "metadata": metadatas[i],
                    "score": float(scores[i])
                })
                
            retriever_logger.info(f"Returning {len(result_docs)} search results from ChromaDB fallback")
            return result_docs
        except Exception as e:
            retriever_logger.error(f"Error searching ChromaDB: {str(e)}", exc_info=True)
            return []
            
    async def save(self, filepath: Optional[str] = None):
        """
        Save the hybrid vector store to disk.
        
        Args:
            filepath: Optional path to save the metadata
            
        Returns:
            Path to the saved metadata
        """
        if filepath is None:
            filepath = str(self.storage_dir / f"hybrid_store_{self.session_id}.json")
            
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save FAISS index to a separate file if it exists
        if self.index is not None:
            index_path = f"{filepath}.index"
            try:
                faiss.write_index(self.index, index_path)
                retriever_logger.info(f"Saved FAISS index to {index_path}")
            except Exception as e:
                retriever_logger.error(f"Error saving FAISS index: {str(e)}", exc_info=True)
        
        # Prepare metadata
        metadata = {
            "session_id": self.session_id,
            "collection_name": self.collection_name,
            "storage_dir": str(self.storage_dir),
            "chroma_dir": str(self.chroma_dir),
            "document_ids": self.document_ids,
            "document_texts": self.document_texts,
            "document_metadata": self.document_metadata,
            "embedding_dim": self.embedding_dim
        }
        
        retriever_logger.info(f"Saving hybrid vector store metadata to {filepath}")
        
        # Save metadata to disk
        try:
            with open(filepath, "w") as f:
                json.dump(metadata, f)
            retriever_logger.info(f"Successfully saved hybrid vector store metadata to {filepath}")
        except Exception as e:
            retriever_logger.error(f"Error saving hybrid vector store metadata: {str(e)}", exc_info=True)
            
        return filepath
        
    @classmethod
    async def load(cls, filepath: str):
        """
        Load a hybrid vector store from metadata.
        
        Args:
            filepath: Path to the saved metadata
            
        Returns:
            A new HybridVectorStore instance
        """
        retriever_logger.info(f"Loading hybrid vector store from: {filepath}")
        
        try:
            # Check if the file exists
            if not os.path.exists(filepath):
                retriever_logger.warning(f"Hybrid vector store file does not exist: {filepath}")
                return cls()
                
            # Load metadata
            with open(filepath, "r") as f:
                metadata = json.load(f)
                
            # Extract necessary information
            session_id = metadata.get("session_id")
            collection_name = metadata.get("collection_name")
            storage_dir = metadata.get("storage_dir")
            chroma_dir = metadata.get("chroma_dir")
            document_ids = metadata.get("document_ids", [])
            document_texts = metadata.get("document_texts", [])
            document_metadata = metadata.get("document_metadata", [])
            embedding_dim = metadata.get("embedding_dim")
            
            if not all([session_id, collection_name, storage_dir, chroma_dir]):
                retriever_logger.warning("Missing required metadata for hybrid vector store")
                return cls()
                
            # Create a new instance with the same storage directory
            store = cls(storage_dir=storage_dir)
            
            # Override the auto-generated session and collection
            store.session_id = session_id
            store.document_ids = document_ids
            store.document_texts = document_texts
            store.document_metadata = document_metadata
            store.embedding_dim = embedding_dim
            
            # Load FAISS index if it exists
            index_path = f"{filepath}.index"
            if os.path.exists(index_path) and embedding_dim is not None:
                try:
                    store.index = faiss.read_index(index_path)
                    retriever_logger.info(f"Loaded FAISS index with {store.index.ntotal} vectors")
                except Exception as e:
                    retriever_logger.error(f"Error loading FAISS index: {str(e)}", exc_info=True)
                    # Initialize an empty index if dimension is known
                    if embedding_dim is not None:
                        store.index = faiss.IndexFlatIP(embedding_dim)
            
            # Connect to the existing ChromaDB collection
            try:
                # Delete the auto-created collection
                store.client.delete_collection(store.collection_name)
                
                # Connect to the existing collection
                store.collection_name = collection_name
                store.collection = store.client.get_collection(name=collection_name)
                
                # Get collection info
                collection_count = store.collection.count()
                retriever_logger.info(f"Loaded ChromaDB collection with {collection_count} documents")
            except Exception as e:
                retriever_logger.error(f"Error connecting to existing ChromaDB collection: {str(e)}", exc_info=True)
                # The auto-created collection will be used as fallback
            
            retriever_logger.info(f"Successfully loaded hybrid vector store")
            return store
        except Exception as e:
            retriever_logger.error(f"Error loading hybrid vector store: {str(e)}", exc_info=True)
            # Return a new store
            return cls() 
```

## .\rag\logger.py

```python
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Create logs directory if it doesn't exist
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Configure the root logger
def setup_logger(name, log_level=logging.INFO):
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Name of the logger
        log_level: Logging level
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Remove existing handlers if any
    if logger.handlers:
        logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # Create file handler
    today = datetime.now().strftime('%Y-%m-%d')
    log_file = logs_dir / f"{name}_{today}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Create loggers for different components
rag_logger = setup_logger("rag")
embedding_logger = setup_logger("embedding")
retriever_logger = setup_logger("retriever")
chat_logger = setup_logger("chat")
files_logger = setup_logger("files") 
```

## .\rag\pipeline.py

```python
from typing import List, Dict, Any, Optional, Union, Literal
import os
from pathlib import Path

from .document_loader import DocumentLoader
from .embedding import EmbeddingGenerator
from .retriever import VectorStore
from .faiss_retriever import FAISSVectorStore
from .chroma_retriever import ChromaVectorStore
from .hybrid_retriever import HybridVectorStore
from .logger import rag_logger


class RAGPipeline:
    """
    Complete RAG pipeline that combines document loading,
    embedding generation, and retrieval.
    """
    
    def __init__(self, vector_store_type: Literal["basic", "faiss", "chroma", "hybrid"] = "basic"):
        """
        Initialize the RAG pipeline with default components.
        
        Args:
            vector_store_type: Type of vector store to use
                - "basic": Simple in-memory vector store
                - "faiss": FAISS-based vector store (faster for large collections)
                - "chroma": ChromaDB-based vector store (persistent with more features)
                - "hybrid": Hybrid implementation combining FAISS and ChromaDB
        """
        self.document_loader = DocumentLoader()
        self.embedding_generator = EmbeddingGenerator()
        
        # Initialize the appropriate vector store
        self.vector_store_type = vector_store_type
        if vector_store_type == "faiss":
            self.vector_store = FAISSVectorStore()
            rag_logger.info("Using FAISS vector store for faster retrieval")
        elif vector_store_type == "chroma":
            self.vector_store = ChromaVectorStore()
            rag_logger.info("Using ChromaDB vector store for persistent storage")
        elif vector_store_type == "hybrid":
            self.vector_store = HybridVectorStore()
            rag_logger.info("Using hybrid vector store (FAISS + ChromaDB) for optimal performance")
        else:
            self.vector_store = VectorStore()
            rag_logger.info("Using basic vector store")
            
        self.last_filepath = None  # Track the filepath the pipeline was loaded from
        
    async def process_file(self, file_path: str) -> int:
        """
        Process a file and add it to the vector store.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Number of document chunks added
        """
        rag_logger.info(f"Processing file: {file_path}")
        
        # Load and chunk the document
        document_chunks = await self.document_loader.load_document(file_path)
        
        rag_logger.info(f"Loaded {len(document_chunks)} document chunks")
        
        if not document_chunks:
            rag_logger.warning("No document chunks found")
            return 0
            
        # Extract the text content for embedding
        texts = [chunk["content"] for chunk in document_chunks]
        
        rag_logger.info(f"Extracted {len(texts)} text chunks for embedding")
        
        # Generate embeddings
        embeddings = await self.embedding_generator.generate_embeddings(texts)
        
        rag_logger.info(f"Generated {len(embeddings)} embeddings")
        
        # Add to the vector store
        await self.vector_store.add_documents(document_chunks, embeddings)
        
        # Return the number of chunks added
        return len(document_chunks)
        
    async def query(self, 
                   query: str, 
                   top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Query the RAG system with a natural language query.
        
        Args:
            query: The user's query
            top_k: Number of top results to return
            
        Returns:
            List of relevant document chunks
        """
        rag_logger.info(f"Querying RAG system with: '{query[:50]}...' (top_k={top_k})")
        
        # Generate embedding for the query
        query_embedding = await self.embedding_generator.generate_single_embedding(query)
        
        # Search the vector store
        results = await self.vector_store.search(query_embedding, top_k)
        
        rag_logger.info(f"Retrieved {len(results)} documents from vector store")
        
        return results
        
    async def save(self, filepath: Optional[str] = None) -> str:
        """
        Save the current state of the RAG pipeline.
        
        Args:
            filepath: Optional path to save to. If None, uses the last loaded path or generates a new one.
            
        Returns:
            Path to the saved vector store
        """
        # Use the provided filepath, or the one we loaded from, or let vector_store generate one
        save_filepath = filepath or self.last_filepath
        filepath = await self.vector_store.save(save_filepath)
        rag_logger.info(f"Saved RAG pipeline to {filepath}")
        self.last_filepath = filepath  # Update the last filepath
        return filepath
        
    @classmethod
    async def load(cls, filepath: str, vector_store_type: Literal["basic", "faiss", "chroma", "hybrid"] = "basic"):
        """
        Load a saved RAG pipeline state.
        
        Args:
            filepath: Path to the saved vector store
            vector_store_type: Type of vector store to load
            
        Returns:
            A new RAGPipeline instance
        """
        rag_logger.info(f"Loading RAG pipeline from: {filepath} with vector store type: {vector_store_type}")
        
        # Create a new instance with the specified vector store type
        pipeline = cls(vector_store_type=vector_store_type)
        pipeline.last_filepath = filepath  # Store the filepath it was loaded from
        
        try:
            # Check if the file exists
            if not os.path.exists(filepath):
                rag_logger.warning(f"Vector store file does not exist: {filepath}")
                return pipeline
                
            # Load the appropriate vector store
            if vector_store_type == "faiss":
                pipeline.vector_store = await FAISSVectorStore.load(filepath)
            elif vector_store_type == "chroma":
                pipeline.vector_store = await ChromaVectorStore.load(filepath)
            elif vector_store_type == "hybrid":
                pipeline.vector_store = await HybridVectorStore.load(filepath)
            else:
                pipeline.vector_store = await VectorStore.load(filepath)
                
        except Exception as e:
            rag_logger.error(f"Error loading vector store: {str(e)}", exc_info=True)
            # Keep the default empty vector store
            
        return pipeline
```

## .\rag\retriever.py

```python
from typing import List, Dict, Any, Optional, Union
import numpy as np
import os
import pickle
from pathlib import Path
import uuid
from .logger import retriever_logger

class VectorStore:
    """
    A simple in-memory vector store for document retrieval.
    
    For production use, consider using a dedicated vector database
    like FAISS, Chroma, or Pinecone.
    """
    
    def __init__(self, storage_dir: str = "vector_store"):
        """
        Initialize the vector store.
        
        Args:
            storage_dir: Directory to save the vector store
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory storage
        self.document_ids = []
        self.document_texts = []
        self.document_metadata = []
        self.embeddings = []
        
        # Unique ID for this session
        self.session_id = str(uuid.uuid4())
        retriever_logger.info(f"Initialized vector store with session ID: {self.session_id}")
        
    async def add_documents(self, 
                           documents: List[Dict[str, Any]], 
                           embeddings: List[List[float]]):
        """
        Add documents and their embeddings to the vector store.
        
        Args:
            documents: List of document dictionaries (content and metadata)
            embeddings: List of embedding vectors for the documents
        """
        retriever_logger.info(f"Adding {len(documents)} documents to vector store")
        retriever_logger.debug(f"Embeddings received: {len(embeddings)} with type {type(embeddings)}")
        
        if len(documents) != len(embeddings):
            retriever_logger.warning(f"Number of documents ({len(documents)}) and embeddings ({len(embeddings)}) do not match")
            raise ValueError("Number of documents and embeddings must match")
            
        # Check if embeddings are valid
        if not embeddings or not isinstance(embeddings[0], list):
            retriever_logger.warning(f"Invalid embeddings format: {type(embeddings)} / First item: {type(embeddings[0]) if embeddings else 'None'}")
            
        # Add each document
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"{self.session_id}_{len(self.document_ids) + i}"
            
            self.document_ids.append(doc_id)
            self.document_texts.append(doc["content"])
            self.document_metadata.append(doc["metadata"])
            self.embeddings.append(embedding)
            
        retriever_logger.info(f"Vector store now contains {len(self.document_ids)} documents and {len(self.embeddings)} embeddings")
            
    async def search(self, 
                    query_embedding: List[float], 
                    top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query embedding.
        
        Args:
            query_embedding: Embedding vector for the query
            top_k: Number of top results to return
            
        Returns:
            List of document dictionaries with similarity scores
        """
        retriever_logger.info(f"Searching vector store with {len(self.embeddings)} embeddings")
        
        if not self.embeddings:
            retriever_logger.warning("No embeddings in vector store")
            return []
            
        # Convert lists to numpy arrays for faster computation
        query_embedding_np = np.array(query_embedding)
        embeddings_np = np.array(self.embeddings)
        
        retriever_logger.debug(f"Query embedding shape: {query_embedding_np.shape}")
        retriever_logger.debug(f"Document embeddings shape: {embeddings_np.shape}")
        
        # Compute cosine similarities
        # First normalize the embeddings
        query_norm = np.linalg.norm(query_embedding_np)
        if query_norm > 0:
            query_embedding_np = query_embedding_np / query_norm
            
        # Normalize document embeddings if they haven't been normalized
        norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
        normalized_embeddings = np.divide(embeddings_np, norms, 
                                         out=np.zeros_like(embeddings_np), 
                                         where=norms!=0)
        
        # Compute cosine similarities (dot product of normalized vectors)
        similarities = np.dot(normalized_embeddings, query_embedding_np)
        
        retriever_logger.debug(f"Computed {len(similarities)} similarity scores")
        
        # Get indices of top-k most similar documents
        if len(similarities) <= top_k:
            top_indices = np.argsort(similarities)[::-1]
        else:
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
        # Prepare results
        results = []
        for idx in top_indices:
            results.append({
                "id": self.document_ids[idx],
                "content": self.document_texts[idx],
                "metadata": self.document_metadata[idx],
                "score": float(similarities[idx])
            })
            
        retriever_logger.info(f"Returning {len(results)} search results")
        for i, result in enumerate(results):
            retriever_logger.debug(f"Result {i+1}: Score {result['score']:.4f}, Content: '{result['content'][:50]}...'")
            
        return results
        
    async def save(self, filepath: Optional[str] = None):
        """
        Save the vector store to disk.
        
        Args:
            filepath: Optional path to save the vector store
        """
        if filepath is None:
            filepath = self.storage_dir / f"vector_store_{self.session_id}.pkl"
            
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
        # Prepare data to save
        data = {
            "document_ids": self.document_ids,
            "document_texts": self.document_texts,
            "document_metadata": self.document_metadata,
            "embeddings": self.embeddings,
            "session_id": self.session_id
        }
        
        retriever_logger.info(f"Saving vector store with {len(self.embeddings)} embeddings to {filepath}")
        
        # Verify data integrity before saving
        if len(self.document_ids) != len(self.embeddings):
            retriever_logger.warning(f"Data integrity issue before saving - document_ids ({len(self.document_ids)}) and embeddings ({len(self.embeddings)}) length mismatch")
        
        # Save to disk
        try:
            with open(filepath, "wb") as f:
                pickle.dump(data, f)
            retriever_logger.info(f"Successfully saved vector store to {filepath}")
        except Exception as e:
            retriever_logger.error(f"Error saving vector store to {filepath}: {str(e)}", exc_info=True)
            
        return filepath
        
    @classmethod
    async def load(cls, filepath: str):
        """
        Load a vector store from disk.
        
        Args:
            filepath: Path to the saved vector store
            
        Returns:
            A new VectorStore instance
        """
        retriever_logger.info(f"Loading vector store from: {filepath}")
        
        try:
            # Check if the file exists
            if not os.path.exists(filepath):
                retriever_logger.warning(f"Vector store file does not exist: {filepath}")
                return cls()
                
            with open(filepath, "rb") as f:
                data = pickle.load(f)
                
            # Create a new instance
            store = cls()
            
            # Validate the data
            required_keys = ["document_ids", "document_texts", "document_metadata", "embeddings", "session_id"]
            for key in required_keys:
                if key not in data:
                    retriever_logger.warning(f"Missing key '{key}' in loaded vector store data")
                    return cls()
            
            # Restore the data
            store.document_ids = data["document_ids"]
            store.document_texts = data["document_texts"]
            store.document_metadata = data["document_metadata"]
            store.embeddings = data["embeddings"]
            store.session_id = data["session_id"]
            
            # Verify the data integrity
            if len(store.document_ids) != len(store.embeddings):
                retriever_logger.warning(f"Data integrity issue - document_ids ({len(store.document_ids)}) and embeddings ({len(store.embeddings)}) length mismatch")
            
            retriever_logger.info(f"Successfully loaded vector store with {len(store.embeddings)} embeddings")
            return store
        except Exception as e:
            retriever_logger.error(f"Error loading vector store from {filepath}: {str(e)}", exc_info=True)
            # Return an empty store
            return cls()
```

## .\rag\__init__.py

```python

```

## .\routes\benchmark.py

```python
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
import asyncio
import logging
from rag.benchmark import RAGBenchmark
from websockets.connection import connection_manager

# Set up logging
benchmark_logger = logging.getLogger("benchmark")

router = APIRouter()

@router.get("/benchmark")
async def run_benchmark(
    user_id: str,
    documents: int = Query(100, ge=10, le=1000),
    queries: int = Query(20, ge=5, le=100),
    batch_size: int = Query(50, ge=10, le=200),
    top_k: int = Query(5, ge=1, le=20),
    implementations: str = Query("basic,faiss,chroma,hybrid")
):
    """
    Run a benchmark of different RAG implementations.
    
    Args:
        user_id: The user's ID
        documents: Number of documents to use in the benchmark
        queries: Number of queries to run
        batch_size: Batch size for adding documents
        top_k: Number of results to retrieve per query
        implementations: Comma-separated list of implementations to benchmark
    """
    try:
        benchmark_logger.info(f"Starting benchmark for user {user_id} with {documents} documents and {queries} queries")
        
        # Parse implementations
        impls = [impl.strip() for impl in implementations.split(",")]
        valid_impls = ["basic", "faiss", "chroma", "hybrid"]
        
        # Validate implementations
        for impl in impls[:]:
            if impl not in valid_impls:
                benchmark_logger.warning(f"Invalid implementation: {impl}")
                impls.remove(impl)
                
        if not impls:
            benchmark_logger.warning("No valid implementations specified")
            raise HTTPException(status_code=400, detail="No valid implementations specified")
            
        benchmark_logger.info(f"Running benchmark with implementations: {', '.join(impls)}")
        
        # Create benchmark instance
        benchmark = RAGBenchmark(vector_store_types=impls)
        
        # Run benchmark in a separate task to avoid blocking
        asyncio.create_task(
            run_benchmark_task(
                user_id=user_id,
                benchmark=benchmark,
                documents=documents,
                queries=queries,
                batch_size=batch_size,
                top_k=top_k
            )
        )
        
        return {
            "status": "started",
            "message": "Benchmark started. Results will be sent via WebSocket when complete."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        benchmark_logger.error(f"Error starting benchmark: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error starting benchmark: {str(e)}")
        
async def run_benchmark_task(
    user_id: str,
    benchmark: RAGBenchmark,
    documents: int,
    queries: int,
    batch_size: int,
    top_k: int
):
    """
    Run the benchmark in a separate task and send results via WebSocket.
    
    Args:
        user_id: The user's ID
        benchmark: The benchmark instance
        documents: Number of documents to use
        queries: Number of queries to run
        batch_size: Batch size for adding documents
        top_k: Number of results to retrieve per query
    """
    try:
        # Run the benchmark
        results = await benchmark.run_benchmark(
            num_documents=documents,
            num_queries=queries,
            batch_size=batch_size,
            top_k=top_k
        )
        
        # Send results to the user
        await connection_manager.send_message(
            user_id,
            {
                "type": "benchmark_result",
                "results": results,
                "timestamp": benchmark_logger.handlers[0].formatter.formatTime(
                    benchmark_logger.makeRecord("benchmark", logging.INFO, "", 0, "", (), None)
                )
            }
        )
        
        benchmark_logger.info(f"Benchmark completed for user {user_id}")
        
    except Exception as e:
        benchmark_logger.error(f"Error running benchmark: {str(e)}", exc_info=True)
        
        # Send error to the user
        await connection_manager.send_message(
            user_id,
            {
                "type": "benchmark_result",
                "results": {"error": str(e)},
                "timestamp": benchmark_logger.handlers[0].formatter.formatTime(
                    benchmark_logger.makeRecord("benchmark", logging.INFO, "", 0, "", (), None)
                )
            }
        ) 
```

## .\routes\chat.py

```python

```

## .\routes\files.py

```python
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional
import os
import uuid
from datetime import datetime
from pathlib import Path

from rag.pipeline import RAGPipeline
from websockets.connection import connection_manager
from rag.logger import files_logger

router = APIRouter()

@router.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: str = Form(...),
    rag_type: str = Form("basic")  # Add RAG type parameter with default
):
    """
    Upload a file for RAG processing.
    
    Args:
        background_tasks: FastAPI background tasks
        file: The uploaded file
        user_id: The user's ID
        rag_type: Type of RAG implementation to use ("basic", "faiss", "chroma", or "hybrid")
    """
    try:
        files_logger.info(f"Received file upload request from user {user_id} with RAG type: {rag_type}")
        
        # Validate RAG type
        if rag_type not in ["basic", "faiss", "chroma", "hybrid"]:
            files_logger.warning(f"Invalid RAG type: {rag_type}, defaulting to 'basic'")
            rag_type = "basic"
        
        # Create uploads directory if it doesn't exist
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        # Generate a unique file ID
        file_id = str(uuid.uuid4())
        
        # Save the file
        file_path = upload_dir / f"{file_id}_{file.filename}"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
            
        files_logger.info(f"File uploaded: {file.filename} (ID: {file_id}) by user {user_id}")
            
        # Process the file in the background
        background_tasks.add_task(
            process_file_background,
            str(file_path),
            user_id,
            file_id,
            file.filename,
            rag_type
        )
        
        return {
            "status": "success",
            "message": "File uploaded and processing started",
            "file_id": file_id,
            "filename": file.filename
        }
    except Exception as e:
        files_logger.error(f"Error uploading file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

async def process_file_background(file_path: str, user_id: str, file_id: str, original_filename: str, rag_type: str = "basic"):
    """
    Process a file in the background and update the RAG pipeline.
    
    Args:
        file_path: Path to the saved file
        user_id: The user's ID
        file_id: The file's ID
        original_filename: Original filename
        rag_type: Type of RAG implementation to use
    """
    try:
        files_logger.info(f"Starting background processing of file {original_filename} (ID: {file_id}) for user {user_id} with RAG type: {rag_type}")
        
        # Initialize or load RAG pipeline for this user
        rag_pipeline = RAGPipeline(vector_store_type=rag_type)
        rag_session = connection_manager.get_user_rag_session(user_id)
        user_rag_type = connection_manager.get_user_rag_type(user_id)
        
        if rag_session:
            try:
                files_logger.info(f"Loading existing RAG session for user {user_id} from path: {rag_session} with type: {user_rag_type or rag_type}")
                rag_pipeline = await RAGPipeline.load(rag_session, vector_store_type=user_rag_type or rag_type)
                files_logger.info(f"RAG session loaded successfully with type: {user_rag_type or rag_type}")
            except Exception as e:
                # If loading fails, use a new pipeline
                files_logger.error(f"Error loading RAG session in process_file_background: {str(e)}", exc_info=True)
                files_logger.info(f"Using a new RAG pipeline with type: {rag_type} instead")
        else:
            files_logger.info(f"No existing RAG session found for user {user_id} in process_file_background, using type: {rag_type}")
        
        # Process the file
        num_chunks = await rag_pipeline.process_file(file_path)
        
        # Save the updated RAG pipeline
        vector_store_path = await rag_pipeline.save()
        files_logger.info(f"Saved RAG pipeline to {vector_store_path}")
        connection_manager.set_user_rag_session(user_id, vector_store_path, rag_type)
        files_logger.info(f"Updated user {user_id} RAG session path to {vector_store_path} with type: {rag_type}")
        
        # Notify the user
        await connection_manager.send_message(
            user_id,
            {
                "type": "file_processed",
                "status": "success",
                "file_id": file_id,
                "filename": original_filename,
                "chunks": num_chunks
            }
        )
    except Exception as e:
        files_logger.error(f"Error processing file {original_filename}: {str(e)}", exc_info=True)
        # Notify the user of the error
        await connection_manager.send_message(
            user_id,
            {
                "type": "file_processed",
                "status": "error",
                "file_id": file_id,
                "filename": original_filename,
                "error": str(e)
            }
        )

@router.post("/switch_rag_type")
async def switch_rag_type(
    user_id: str = Form(...),
    rag_type: str = Form(...)
):
    """
    Switch the RAG implementation type for a user.
    
    Args:
        user_id: The user's ID
        rag_type: Type of RAG implementation to use ("basic", "faiss", "chroma", or "hybrid")
    """
    try:
        files_logger.info(f"Switching RAG type for user {user_id} to {rag_type}")
        
        # Validate RAG type
        if rag_type not in ["basic", "faiss", "chroma", "hybrid"]:
            files_logger.warning(f"Invalid RAG type: {rag_type}")
            raise HTTPException(status_code=400, detail=f"Invalid RAG type: {rag_type}. Must be one of: basic, faiss, chroma, hybrid")
        
        # Get the current RAG session
        current_session = connection_manager.get_user_rag_session(user_id)
        current_type = connection_manager.get_user_rag_type(user_id)
        
        if not current_session:
            # No existing session, just set the type preference
            connection_manager.set_user_rag_session(user_id, None, rag_type)
            return {"status": "success", "message": f"RAG type set to {rag_type} for future sessions"}
        
        # Load the current RAG pipeline
        try:
            current_pipeline = await RAGPipeline.load(current_session, vector_store_type=current_type or "basic")
            
            # Create a new pipeline with the desired type
            new_pipeline = RAGPipeline(vector_store_type=rag_type)
            
            # If the current pipeline has documents, we need to transfer them
            # This is a simplified approach - in a real implementation, you might want to
            # process the documents again to optimize for the new vector store type
            
            # For simplicity, we'll just process the same files again if they exist
            # A more sophisticated approach would be to extract and transfer the embeddings directly
            
            # Save the new pipeline
            new_session_path = await new_pipeline.save()
            
            # Update the user's session
            connection_manager.set_user_rag_session(user_id, new_session_path, rag_type)
            
            return {
                "status": "success", 
                "message": f"Switched RAG type from {current_type or 'basic'} to {rag_type}",
                "note": "You may need to re-upload your documents for optimal performance with the new RAG type"
            }
            
        except Exception as e:
            files_logger.error(f"Error switching RAG type: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error switching RAG type: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        files_logger.error(f"Error switching RAG type: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
```

## .\routes\__init__.py

```python

```

## .\websockets\chat.py

```python
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from typing import Dict, List, Optional, Any
import json
import uuid
import asyncio
from datetime import datetime

from models.base import BaseLanguageModel
from models.huggingface import HuggingFaceModel
from models.local_model import LocalModel
from models.lm_studio import LMStudioModel
from rag.pipeline import RAGPipeline
from .connection import connection_manager
from rag.logger import chat_logger

router = APIRouter()

# Model factory function
def get_model(model_type: str) -> BaseLanguageModel:
    """
    Get the appropriate model based on the type.
    
    Args:
        model_type: Type of model to use
        
    Returns:
        A language model instance
    """
    if model_type == "huggingface":
        return HuggingFaceModel()
    elif model_type == "local":
        return LocalModel()
    elif model_type == "lm_studio":
        return LMStudioModel()
    else:
        # Default to local model
        return LocalModel()

@router.websocket("/ws/chat/{user_id}")
async def websocket_chat(websocket: WebSocket, user_id: str):
    """
    WebSocket endpoint for chat.
    
    Args:
        websocket: The WebSocket connection
        user_id: The user's ID
    """
    await connection_manager.connect(websocket, user_id)
    chat_logger.info(f"User {user_id} connected to chat websocket")
    
    # Start a heartbeat task to keep the connection alive
    heartbeat_task = asyncio.create_task(send_heartbeat(user_id))
    
    try:
        # Send initial connection acknowledgment
        await connection_manager.send_message(
            user_id, 
            {
                "type": "connection_status",
                "status": "connected",
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Chat loop
        while True:
            # Wait for messages
            data = await websocket.receive_text()
            
            try:
                # Parse the message
                message_data = json.loads(data)
                
                # Process the message
                if message_data.get("type") == "message":
                    chat_logger.info(f"Received chat message from user {user_id}")
                    # Process message in a separate task to avoid blocking
                    asyncio.create_task(process_chat_message(user_id, message_data))
                elif message_data.get("type") == "ping":
                    # Respond to ping with a pong to keep the connection alive
                    chat_logger.debug(f"Received ping from user {user_id}")
                    await connection_manager.send_message(
                        user_id,
                        {
                            "type": "pong",
                            "timestamp": datetime.now().isoformat()
                        }
                    )
                    
            except json.JSONDecodeError:
                chat_logger.error(f"Invalid JSON received from user {user_id}: {data}")
                await connection_manager.send_message(
                    user_id,
                    {
                        "type": "error",
                        "error": "Invalid message format",
                        "timestamp": datetime.now().isoformat()
                    }
                )
                
    except WebSocketDisconnect:
        chat_logger.info(f"User {user_id} disconnected from chat websocket")
        connection_manager.disconnect(user_id)
        # Cancel the heartbeat task
        heartbeat_task.cancel()
    except Exception as e:
        chat_logger.error(f"Error in websocket chat: {str(e)}", exc_info=True)
        # Cancel the heartbeat task
        heartbeat_task.cancel()


async def send_heartbeat(user_id: str):
    """
    Send periodic heartbeat messages to keep the connection alive.
    
    Args:
        user_id: The user's ID
    """
    try:
        while True:
            # Send a heartbeat every 15 seconds
            await asyncio.sleep(15)
            if user_id in connection_manager.active_connections:
                await connection_manager.send_message(
                    user_id,
                    {
                        "type": "heartbeat",
                        "timestamp": datetime.now().isoformat()
                    }
                )
    except asyncio.CancelledError:
        # Task was cancelled, exit gracefully
        pass
    except Exception as e:
        chat_logger.error(f"Error in heartbeat task: {str(e)}", exc_info=True)


async def process_chat_message(user_id: str, data: Dict):
    """
    Process a chat message and generate a response.
    
    Args:
        user_id: The user's ID
        data: The message data
    """
    # Extract message content
    prompt = data.get("content", "")
    if not prompt:
        chat_logger.warning(f"Empty prompt received from user {user_id}")
        return
        
    # Extract options
    use_rag = data.get("use_rag", False)
    model_type = data.get("model_type", "local")
    system_prompt = data.get("system_prompt", "You are a helpful assistant.")
    temperature = data.get("temperature", 0.7)
    max_tokens = data.get("max_tokens", 1024)
    rag_type = data.get("rag_type", "basic")  # Get RAG type from request or use default
    
    chat_logger.info(f"Processing chat message from user {user_id}: '{prompt[:50]}...' (use_rag={use_rag}, model={model_type}, rag_type={rag_type})")
    
    # Initialize RAG pipeline for this specific request if needed
    rag_pipeline = None
    
    if use_rag:
        # Load the RAG pipeline from the user's session
        rag_session = connection_manager.get_user_rag_session(user_id)
        user_rag_type = connection_manager.get_user_rag_type(user_id)
        
        # Use the user's stored RAG type if available, otherwise use the one from the request
        rag_type = user_rag_type if user_rag_type else rag_type
        
        if rag_session:
            try:
                chat_logger.info(f"Loading RAG session for user {user_id} from path: {rag_session} with type: {rag_type}")
                rag_pipeline = await RAGPipeline.load(rag_session, vector_store_type=rag_type)
                chat_logger.info(f"Loaded RAG pipeline with type: {rag_type}")
            except Exception as e:
                chat_logger.error(f"Error loading RAG session: {str(e)}", exc_info=True)
                # Create a new pipeline if loading fails
                rag_pipeline = RAGPipeline(vector_store_type=rag_type)
        else:
            # Create a new pipeline if no session exists
            chat_logger.info(f"No RAG session found for user {user_id}, creating new pipeline with type: {rag_type}")
            rag_pipeline = RAGPipeline(vector_store_type=rag_type)
    
    # Get conversation history (if provided)
    context = data.get("context", [])
    
    # Get the appropriate model
    try:
        model = get_model(model_type)
    except Exception as e:
        error_message = f"Error initializing {model_type} model: {str(e)}"
        chat_logger.error(error_message, exc_info=True)
        await connection_manager.send_message(
            user_id,
            {
                "type": "error",
                "error": error_message,
                "timestamp": datetime.now().isoformat()
            }
        )
        return
    
    # Send typing indicator
    await connection_manager.send_message(
        user_id,
        {
            "type": "typing",
            "status": "started",
            "timestamp": datetime.now().isoformat()
        }
    )
    
    try:
        # Generate response
        sources = []
        if use_rag and rag_pipeline:
            # Query the RAG system
            chat_logger.info(f"Using RAG ({rag_type}) for query: '{prompt[:50]}...'")
            
            # Check if vector store has documents (different check for each type)
            has_documents = False
            if rag_type == "basic" and hasattr(rag_pipeline.vector_store, 'embeddings'):
                has_documents = len(rag_pipeline.vector_store.embeddings) > 0
            elif rag_type == "faiss" and rag_pipeline.vector_store.index is not None:
                has_documents = rag_pipeline.vector_store.index.ntotal > 0
            elif rag_type == "chroma":
                try:
                    has_documents = rag_pipeline.vector_store.collection.count() > 0
                except:
                    has_documents = False
            
            # Skip RAG if no documents are available
            if not has_documents:
                chat_logger.warning(f"No documents available in the {rag_type} vector store, falling back to standard response")
                await connection_manager.send_message(
                    user_id,
                    {
                        "type": "warning",
                        "message": f"No document embeddings found in {rag_type} store. Using standard response instead.",
                        "timestamp": datetime.now().isoformat()
                    }
                )
                response = await model.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    context=context
                )
            else:
                try:
                    retrieved_docs = await rag_pipeline.query(prompt, top_k=5)
                    chat_logger.info(f"Retrieved {len(retrieved_docs)} documents from {rag_type} RAG")
                    
                    # Extract the content from retrieved documents
                    doc_contents = [doc["content"] for doc in retrieved_docs]
                    
                    # Log retrieved documents
                    for i, doc in enumerate(retrieved_docs):
                        chat_logger.debug(f"Retrieved document {i+1}: Score {doc['score']:.4f}, Content: '{doc['content'][:50]}...'")
                    
                    # Generate response with RAG context
                    chat_logger.info(f"Generating response with RAG using {model_type} model")
                    response = await model.generate_with_rag(
                        prompt=prompt,
                        documents=doc_contents,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        context=context
                    )
                    
                    # Check if response is valid
                    if not response or not isinstance(response, str):
                        raise ValueError(f"Invalid response from {model_type} model with RAG: {response}")
                    
                    # Include retrieved documents in response metadata
                    sources = [
                        {
                            "content": doc["content"],
                            "metadata": doc["metadata"],
                            "score": doc["score"]
                        }
                        for doc in retrieved_docs
                    ]
                except Exception as e:
                    chat_logger.error(f"Error in RAG processing with {model_type} model: {str(e)}", exc_info=True)
                    # Fallback to non-RAG response
                    response = await model.generate(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        context=context
                    )
                    await connection_manager.send_message(
                        user_id,
                        {
                            "type": "warning",
                            "message": f"Error using RAG with {model_type} model. Falling back to standard response.",
                            "timestamp": datetime.now().isoformat()
                        }
                    )
        else:
            # Generate response without RAG
            chat_logger.info(f"Generating response without RAG using {model_type} model")
            response = await model.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                context=context
            )
            
        chat_logger.info(f"Generated response for user {user_id}: '{response[:50]}...'")
            
        # Send the response
        await connection_manager.send_message(
            user_id,
            {
                "type": "message",
                "content": response,
                "role": "assistant",
                "id": f"assistant-{uuid.uuid4()}",
                "metadata": {"sources": sources} if sources else {},
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Save the RAG pipeline if it was used
        if use_rag and rag_pipeline:
            try:
                # Save to the user's RAG session
                vector_store_path = await rag_pipeline.save()
                connection_manager.set_user_rag_session(user_id, vector_store_path)
                chat_logger.info(f"Saved RAG pipeline to {vector_store_path} with {len(rag_pipeline.vector_store.embeddings)} embeddings")
            except Exception as e:
                chat_logger.error(f"Error saving RAG session: {str(e)}", exc_info=True)
            
    except Exception as e:
        # Send error message
        error_message = f"Error generating response: {str(e)}"
        chat_logger.error(error_message, exc_info=True)
        await connection_manager.send_message(
            user_id,
            {
                "type": "error",
                "error": error_message,
                "timestamp": datetime.now().isoformat()
            }
        )
    finally:
        # Stop typing indicator
        await connection_manager.send_message(
            user_id,
            {
                "type": "typing",
                "status": "stopped",
                "timestamp": datetime.now().isoformat()
            }
        )
```

## .\websockets\connection.py

```python
from typing import Dict, List, Set, Optional
from fastapi import WebSocket
import json
import asyncio

class ConnectionManager:
    """
    Manager for WebSocket connections.
    """
    
    def __init__(self):
        """
        Initialize the connection manager.
        """
        self.active_connections = {}
        self.user_rag_sessions = {}
        self.user_rag_types = {}  # Track the RAG type for each user
        
    async def connect(self, websocket: WebSocket, user_id: str):
        """
        Connect a new WebSocket and store it with the user ID.
        
        Args:
            websocket: The WebSocket connection
            user_id: The user's ID
        """
        await websocket.accept()
        self.active_connections[user_id] = websocket
        
    def disconnect(self, user_id: str):
        """
        Disconnect a WebSocket by user ID.
        
        Args:
            user_id: The user's ID
        """
        if user_id in self.active_connections:
            del self.active_connections[user_id]
            
    async def send_message(self, user_id: str, message: Dict):
        """
        Send a message to a specific user.
        
        Args:
            user_id: The user's ID
            message: The message to send
        """
        if user_id in self.active_connections:
            await self.active_connections[user_id].send_json(message)
            
    async def broadcast(self, message: Dict):
        """
        Broadcast a message to all connected users.
        
        Args:
            message: The message to broadcast
        """
        for connection in self.active_connections.values():
            await connection.send_json(message)
            
    def set_user_rag_session(self, user_id: str, vector_store_path: str, rag_type: str = "basic"):
        """
        Set the RAG session for a user.
        
        Args:
            user_id: The user's ID
            vector_store_path: Path to the user's vector store
            rag_type: Type of RAG implementation ("basic", "faiss", or "chroma")
        """
        self.user_rag_sessions[user_id] = vector_store_path
        self.user_rag_types[user_id] = rag_type
        
    def get_user_rag_session(self, user_id: str) -> Optional[str]:
        """
        Get the RAG session for a user.
        
        Args:
            user_id: The user's ID
            
        Returns:
            Path to the user's vector store, if any
        """
        return self.user_rag_sessions.get(user_id)
        
    def get_user_rag_type(self, user_id: str) -> str:
        """
        Get the RAG type for a user.
        
        Args:
            user_id: The user's ID
            
        Returns:
            Type of RAG implementation for the user ("basic", "faiss", or "chroma")
        """
        return self.user_rag_types.get(user_id, "basic")


# Create a global instance
connection_manager = ConnectionManager()
```

## .\websockets\__init__.py

```python

```

