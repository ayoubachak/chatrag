from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
import asyncio
import logging
from rag.benchmark import RAGBenchmark
from websockets.connection import connection_manager

# Set up logging
benchmark_logger = logging.getLogger("benchmark")

router = APIRouter()

@router.get("/benchmark")
async def run_benchmark(
    user_id: str,
    documents: int = Query(100, ge=10, le=1000),
    queries: int = Query(20, ge=5, le=100),
    batch_size: int = Query(50, ge=10, le=200),
    top_k: int = Query(5, ge=1, le=20),
    implementations: str = Query("basic,faiss,chroma,hybrid")
):
    """
    Run a benchmark of different RAG implementations.
    
    Args:
        user_id: The user's ID
        documents: Number of documents to use in the benchmark
        queries: Number of queries to run
        batch_size: Batch size for adding documents
        top_k: Number of results to retrieve per query
        implementations: Comma-separated list of implementations to benchmark
    """
    try:
        benchmark_logger.info(f"Starting benchmark for user {user_id} with {documents} documents and {queries} queries")
        
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
        
        # Run benchmark in a separate task to avoid blocking
        asyncio.create_task(
            run_benchmark_task(
                user_id=user_id,
                benchmark=benchmark,
                documents=documents,
                queries=queries,
                batch_size=batch_size,
                top_k=top_k
            )
        )
        
        return {
            "status": "started",
            "message": "Benchmark started. Results will be sent via WebSocket when complete."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        benchmark_logger.error(f"Error starting benchmark: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error starting benchmark: {str(e)}")
        
async def run_benchmark_task(
    user_id: str,
    benchmark: RAGBenchmark,
    documents: int,
    queries: int,
    batch_size: int,
    top_k: int
):
    """
    Run the benchmark in a separate task and send results via WebSocket.
    
    Args:
        user_id: The user's ID
        benchmark: The benchmark instance
        documents: Number of documents to use
        queries: Number of queries to run
        batch_size: Batch size for adding documents
        top_k: Number of results to retrieve per query
    """
    try:
        # Run the benchmark
        results = await benchmark.run_benchmark(
            num_documents=documents,
            num_queries=queries,
            batch_size=batch_size,
            top_k=top_k
        )
        
        # Send results to the user
        await connection_manager.send_message(
            user_id,
            {
                "type": "benchmark_result",
                "results": results,
                "timestamp": benchmark_logger.handlers[0].formatter.formatTime(
                    benchmark_logger.makeRecord("benchmark", logging.INFO, "", 0, "", (), None)
                )
            }
        )
        
        benchmark_logger.info(f"Benchmark completed for user {user_id}")
        
    except Exception as e:
        benchmark_logger.error(f"Error running benchmark: {str(e)}", exc_info=True)
        
        # Send error to the user
        await connection_manager.send_message(
            user_id,
            {
                "type": "benchmark_result",
                "results": {"error": str(e)},
                "timestamp": benchmark_logger.handlers[0].formatter.formatTime(
                    benchmark_logger.makeRecord("benchmark", logging.INFO, "", 0, "", (), None)
                )
            }
        ) 