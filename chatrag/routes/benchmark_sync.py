from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional, Dict, Any
import logging
import datetime
import os
import json
import traceback
from rag.benchmark import RAGBenchmark

# Set up logging with proper handler
benchmark_logger = logging.getLogger("benchmark")
if not benchmark_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    benchmark_logger.addHandler(handler)
    benchmark_logger.setLevel(logging.INFO)

router = APIRouter()

@router.get("/benchmark-sync")
async def run_benchmark_sync(
    documents: int = Query(100, ge=10, le=1000),
    queries: int = Query(20, ge=5, le=100),
    batch_size: int = Query(50, ge=10, le=200),
    top_k: int = Query(5, ge=1, le=20),
    implementations: str = Query("basic,faiss,chroma,hybrid")
):
    """
    Run a benchmark of different RAG implementations synchronously.
    
    Args:
        documents: Number of documents to use in the benchmark
        queries: Number of queries to run
        batch_size: Batch size for adding documents
        top_k: Number of results to retrieve per query
        implementations: Comma-separated list of implementations to benchmark
    """
    try:
        # Set OpenMP environment variable to suppress FAISS warnings
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        
        benchmark_logger.info(f"Starting synchronous benchmark with {documents} documents and {queries} queries")
        
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
        
        # Run benchmark synchronously
        results = await benchmark.run_benchmark(
            num_documents=documents,
            num_queries=queries,
            batch_size=batch_size,
            top_k=top_k
        )
        
        if not results:
            benchmark_logger.error("Benchmark returned empty results")
            raise HTTPException(status_code=500, detail="Benchmark returned empty results")
        
        # Make sure results are JSON serializable
        serializable_results = make_serializable(results)
        
        benchmark_logger.info(f"Synchronous benchmark completed successfully")
        
        return serializable_results
        
    except HTTPException:
        raise
    except Exception as e:
        error_details = traceback.format_exc()
        benchmark_logger.error(f"Error in synchronous benchmark: {str(e)}\n{error_details}")
        
        # Attempt to create a minimal fallback response
        fallback_response = {
            "error": str(e),
            "status": "failed",
            "message": "Benchmark failed to complete. Check server logs for details."
        }
        
        # Return a 500 error with detailed explanation
        raise HTTPException(status_code=500, detail=fallback_response)

def make_serializable(obj: Any) -> Any:
    """
    Convert any object to a JSON serializable format.
    
    Args:
        obj: The object to convert
        
    Returns:
        A JSON serializable version of the object
    """
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        try:
            # Try to convert to a basic type
            return str(obj)
        except:
            return f"<non-serializable object of type {type(obj).__name__}>"