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