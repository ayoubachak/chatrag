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