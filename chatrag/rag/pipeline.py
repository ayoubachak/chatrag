from typing import List, Dict, Any, Optional, Union
import os
from pathlib import Path

from .document_loader import DocumentLoader
from .embedding import EmbeddingGenerator
from .retriever import VectorStore
from .logger import rag_logger


class RAGPipeline:
    """
    Complete RAG pipeline that combines document loading,
    embedding generation, and retrieval.
    """
    
    def __init__(self):
        """
        Initialize the RAG pipeline with default components.
        """
        self.document_loader = DocumentLoader()
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = VectorStore()
        
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
        
        rag_logger.info(f"Added documents to vector store. Current size: {len(self.vector_store.embeddings)}")
        
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
        
    async def save(self) -> str:
        """
        Save the current state of the RAG pipeline.
        
        Returns:
            Path to the saved vector store
        """
        filepath = await self.vector_store.save()
        rag_logger.info(f"Saved RAG pipeline to {filepath} with {len(self.vector_store.embeddings)} embeddings")
        return filepath
        
    @classmethod
    async def load(cls, filepath: str):
        """
        Load a saved RAG pipeline state.
        
        Args:
            filepath: Path to the saved vector store
            
        Returns:
            A new RAGPipeline instance
        """
        rag_logger.info(f"Loading RAG pipeline from: {filepath}")
        
        # Create a new instance
        pipeline = cls()
        
        try:
            # Check if the file exists
            if not os.path.exists(filepath):
                rag_logger.warning(f"Vector store file does not exist: {filepath}")
                return pipeline
                
            # Load the vector store
            pipeline.vector_store = await VectorStore.load(filepath)
            rag_logger.info(f"Loaded vector store with {len(pipeline.vector_store.embeddings)} embeddings")
            
            # Verify the vector store has embeddings
            if not pipeline.vector_store.embeddings:
                rag_logger.warning(f"Loaded vector store has no embeddings")
                
        except Exception as e:
            rag_logger.error(f"Error loading vector store: {str(e)}", exc_info=True)
            # Keep the default empty vector store
            
        return pipeline