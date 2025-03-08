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