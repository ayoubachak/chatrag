from typing import List, Dict, Any, Optional, Union, Literal
import os
from pathlib import Path

from .document_loader import DocumentLoader
from .embedding import EmbeddingGenerator
from .base_retriever import BaseVectorStore
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
    
    def __init__(self, vector_store_type: Literal["basic", "faiss", "chroma", "hybrid"] = "basic", chunking_strategy: Literal["basic", "super"] = "basic"):
        """
        Initialize the RAG pipeline with default components.
        
        Args:
            vector_store_type: Type of vector store to use
                - "basic": Simple in-memory vector store
                - "faiss": FAISS-based vector store (faster for large collections)
                - "chroma": ChromaDB-based vector store (persistent with more features)
                - "hybrid": Hybrid implementation combining FAISS and ChromaDB
            chunking_strategy: Strategy to use for chunking documents
                - "basic": Simple chunking by paragraphs and pages
                - "super": Advanced semantic chunking with overlap
        """
        self.document_loader = DocumentLoader(chunking_strategy=chunking_strategy)
        self.embedding_generator = EmbeddingGenerator()
        self.chunking_strategy = chunking_strategy
        
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
            
        rag_logger.info(f"Using {chunking_strategy} chunking strategy")
        self.last_filepath = None  # Track the filepath the pipeline was loaded from
        
    def set_chunking_strategy(self, strategy: Literal["basic", "super"]):
        """
        Change the chunking strategy.
        
        Args:
            strategy: The chunking strategy to use
        """
        if strategy != self.chunking_strategy:
            self.chunking_strategy = strategy
            self.document_loader.set_chunking_strategy(strategy)
            rag_logger.info(f"Changed to {strategy} chunking strategy")
        
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
    async def load(cls, filepath: str, vector_store_type: Literal["basic", "faiss", "chroma", "hybrid"] = "basic", chunking_strategy: Literal["basic", "super"] = "basic"):
        """
        Load a saved RAG pipeline state.
        
        Args:
            filepath: Path to the saved vector store
            vector_store_type: Type of vector store to load
            chunking_strategy: Strategy to use for chunking documents
            
        Returns:
            A new RAGPipeline instance
        """
        rag_logger.info(f"Loading RAG pipeline from: {filepath} with vector store type: {vector_store_type} and chunking strategy: {chunking_strategy}")
        
        # Create a new instance with the specified vector store type and chunking strategy
        pipeline = cls(vector_store_type=vector_store_type, chunking_strategy=chunking_strategy)
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