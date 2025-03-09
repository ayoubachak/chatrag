from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
import os
from pathlib import Path
import uuid

class BaseVectorStore(ABC):
    """
    Abstract base class for vector stores.
    
    All vector store implementations should inherit from this class
    and implement its abstract methods.
    """
    
    def __init__(self, storage_dir: str = "vector_store"):
        """
        Initialize the vector store.
        
        Args:
            storage_dir: Directory to save the vector store
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Unique ID for this session
        self.session_id = str(uuid.uuid4())
        
    @abstractmethod
    async def add_documents(self, 
                           documents: List[Dict[str, Any]], 
                           embeddings: List[List[float]]):
        """
        Add documents and their embeddings to the vector store.
        
        Args:
            documents: List of document dictionaries (content and metadata)
            embeddings: List of embedding vectors for the documents
        """
        pass
            
    @abstractmethod
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
        pass
        
    @abstractmethod
    async def save(self, filepath: Optional[str] = None) -> str:
        """
        Save the vector store to disk.
        
        Args:
            filepath: Optional path to save the vector store
            
        Returns:
            Path to the saved vector store
        """
        pass
        
    @classmethod
    @abstractmethod
    async def load(cls, filepath: str):
        """
        Load a vector store from disk.
        
        Args:
            filepath: Path to the saved vector store
            
        Returns:
            A new vector store instance
        """
        pass
        
    @property
    @abstractmethod
    def document_count(self) -> int:
        """
        Get the number of documents in the vector store.
        
        Returns:
            Number of documents
        """
        pass 