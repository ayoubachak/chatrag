from typing import List, Dict, Any, Optional, Union
import numpy as np
import os
import pickle
import faiss
from pathlib import Path
import uuid
from .logger import retriever_logger
from .base_retriever import BaseVectorStore

class FAISSVectorStore(BaseVectorStore):
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
        super().__init__(storage_dir)
        
        # In-memory storage for document data
        self.document_ids = []
        self.document_texts = []
        self.document_metadata = []
        
        # FAISS index will be initialized when first embeddings are added
        self.index = None
        self.embedding_dim = None
        
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
            
        # Convert query embedding to numpy array and normalize
        query_np = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_np)
        
        # Search the FAISS index
        scores, indices = self.index.search(query_np, min(top_k, len(self.document_ids)))
        
        # Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.document_ids):
                # Skip invalid indices (can happen with some FAISS index types)
                continue
                
            results.append({
                "id": self.document_ids[idx],
                "content": self.document_texts[idx],
                "metadata": self.document_metadata[idx],
                "score": float(scores[0][i])
            })
            
        retriever_logger.info(f"Returning {len(results)} search results from FAISS vector store")
        for i, result in enumerate(results):
            retriever_logger.debug(f"Result {i+1}: Score {result['score']:.4f}, Content: '{result['content'][:50]}...'")
            
        return results
        
    async def save(self, filepath: Optional[str] = None) -> str:
        """
        Save the FAISS vector store to disk.
        
        Args:
            filepath: Optional path to save the vector store
            
        Returns:
            Path to the saved vector store
        """
        if filepath is None:
            filepath = self.storage_dir / f"faiss_store_{self.session_id}.pkl"
            
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Prepare data to save
        data = {
            "document_ids": self.document_ids,
            "document_texts": self.document_texts,
            "document_metadata": self.document_metadata,
            "embedding_dim": self.embedding_dim,
            "session_id": self.session_id
        }
        
        retriever_logger.info(f"Saving FAISS vector store with {len(self.document_ids)} documents to {filepath}")
        
        try:
            # Save the FAISS index separately if it exists
            if self.index is not None:
                index_path = str(filepath) + ".index"
                faiss.write_index(self.index, index_path)
                data["index_path"] = index_path
                
            # Save the metadata
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
                
            # Load the metadata
            with open(filepath, "rb") as f:
                data = pickle.load(f)
                
            # Create a new instance
            store = cls()
            
            # Restore the metadata
            store.document_ids = data.get("document_ids", [])
            store.document_texts = data.get("document_texts", [])
            store.document_metadata = data.get("document_metadata", [])
            store.embedding_dim = data.get("embedding_dim")
            store.session_id = data.get("session_id", str(uuid.uuid4()))
            
            # Load the FAISS index if it exists
            index_path = data.get("index_path")
            if index_path and os.path.exists(index_path):
                store.index = faiss.read_index(index_path)
                retriever_logger.info(f"Loaded FAISS index with {store.index.ntotal} vectors")
            else:
                retriever_logger.warning("No FAISS index found or index path is invalid")
                
            retriever_logger.info(f"Successfully loaded FAISS vector store with {len(store.document_ids)} documents")
            return store
        except Exception as e:
            retriever_logger.error(f"Error loading FAISS vector store from {filepath}: {str(e)}", exc_info=True)
            # Return an empty store
            return cls()
            
    @property
    def document_count(self) -> int:
        """
        Get the number of documents in the vector store.
        
        Returns:
            Number of documents
        """
        return len(self.document_ids) 