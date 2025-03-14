from typing import List, Dict, Any, Optional, Union
import numpy as np
import os
import pickle
from pathlib import Path
import uuid
from .logger import retriever_logger
from .base_retriever import BaseVectorStore

class VectorStore(BaseVectorStore):
    """
    A simple in-memory vector store for document retrieval.
    """
    
    def __init__(self, storage_dir: str = "vector_store"):
        """
        Initialize the vector store.
        
        Args:
            storage_dir: Directory to save the vector store
        """
        super().__init__(storage_dir)
        
        # In-memory storage
        self.document_ids = []
        self.document_texts = []
        self.document_metadata = []
        self.embeddings = []
        
        retriever_logger.info(f"Initialized basic vector store with session ID: {self.session_id}")
        
    async def add_documents(self, 
                           documents: List[Dict[str, Any]], 
                           embeddings: List[List[float]]):
        """
        Add documents and their embeddings to the vector store.
        
        Args:
            documents: List of document dictionaries (content and metadata)
            embeddings: List of embedding vectors for the documents
        """
        retriever_logger.info(f"Adding {len(documents)} documents to basic vector store")
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
            
        retriever_logger.info(f"Basic vector store now contains {len(self.document_ids)} documents and {len(self.embeddings)} embeddings")
            
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
        retriever_logger.info(f"Searching basic vector store with {len(self.embeddings)} embeddings")
        
        if not self.embeddings:
            retriever_logger.warning("No embeddings in basic vector store")
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
            
        retriever_logger.info(f"Returning {len(results)} search results from basic vector store")
        for i, result in enumerate(results):
            retriever_logger.debug(f"Result {i+1}: Score {result['score']:.4f}, Content: '{result['content'][:50]}...'")
            
        return results
        
    async def save(self, filepath: Optional[str] = None) -> str:
        """
        Save the vector store to disk.
        
        Args:
            filepath: Optional path to save the vector store
            
        Returns:
            Path to the saved vector store
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
        
        retriever_logger.info(f"Saving basic vector store with {len(self.embeddings)} embeddings to {filepath}")
        
        # Verify data integrity before saving
        if len(self.document_ids) != len(self.embeddings):
            retriever_logger.warning(f"Data integrity issue before saving - document_ids ({len(self.document_ids)}) and embeddings ({len(self.embeddings)}) length mismatch")
        
        # Save to disk
        try:
            with open(filepath, "wb") as f:
                pickle.dump(data, f)
            retriever_logger.info(f"Successfully saved basic vector store to {filepath}")
        except Exception as e:
            retriever_logger.error(f"Error saving basic vector store to {filepath}: {str(e)}", exc_info=True)
            
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
        retriever_logger.info(f"Loading basic vector store from: {filepath}")
        
        try:
            # Check if the file exists
            if not os.path.exists(filepath):
                retriever_logger.warning(f"Basic vector store file does not exist: {filepath}")
                return cls()
                
            with open(filepath, "rb") as f:
                data = pickle.load(f)
                
            # Create a new instance
            store = cls()
            
            # Validate the data
            required_keys = ["document_ids", "document_texts", "document_metadata", "embeddings", "session_id"]
            for key in required_keys:
                if key not in data:
                    retriever_logger.warning(f"Missing key '{key}' in loaded basic vector store data")
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
            
            retriever_logger.info(f"Successfully loaded basic vector store with {len(store.embeddings)} embeddings")
            return store
        except Exception as e:
            retriever_logger.error(f"Error loading basic vector store from {filepath}: {str(e)}", exc_info=True)
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