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