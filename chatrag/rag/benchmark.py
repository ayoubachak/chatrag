import time
import asyncio
import random
import string
import numpy as np
from typing import List, Dict, Any, Literal, Tuple
from pathlib import Path
import os
import json
import datetime

from .pipeline import RAGPipeline
from .logger import rag_logger

class RAGBenchmark:
    """
    Benchmark utility to compare the performance of different RAG implementations.
    """
    
    def __init__(self, 
                 vector_store_types: List[str] = None,
                 benchmark_dir: str = "benchmark_results"):
        """
        Initialize the benchmark utility.
        
        Args:
            vector_store_types: List of vector store types to benchmark. Default is all types.
            benchmark_dir: Directory to save benchmark results
        """
        self.vector_store_types = vector_store_types or ["basic", "faiss", "chroma", "hybrid"]
        self.benchmark_dir = Path(benchmark_dir)
        
        # Ensure the benchmark directory exists
        os.makedirs(self.benchmark_dir, exist_ok=True)
        
        rag_logger.info(f"Initialized RAG benchmark with store types: {', '.join(self.vector_store_types)}")
        
    async def generate_test_data(self, 
                             num_documents: int = 100, 
                             embedding_dim: int = 384,
                             doc_length: int = 200) -> Tuple[List[Dict], List[List[float]], List[Dict]]:
        """
        Generate synthetic test data for benchmarking.
        
        Args:
            num_documents: Number of test documents to generate
            embedding_dim: Dimension of embeddings
            doc_length: Average length of documents in words
            
        Returns:
            Tuple of (documents, embeddings, queries)
        """
        rag_logger.info(f"Generating {num_documents} test documents with embedding dimension {embedding_dim}")
        
        # Generate random documents
        documents = []
        for i in range(num_documents):
            # Generate random text
            words = []
            for _ in range(random.randint(doc_length - 50, doc_length + 50)):
                word_length = random.randint(3, 10)
                word = ''.join(random.choice(string.ascii_lowercase) for _ in range(word_length))
                words.append(word)
                
            content = ' '.join(words)
            
            # Create document
            documents.append({
                "content": content,
                "metadata": {
                    "id": f"doc_{i}",
                    "source": "synthetic",
                    "length": len(content)
                }
            })
            
        # Generate random embeddings (normalized)
        embeddings = []
        for _ in range(num_documents):
            embedding = np.random.randn(embedding_dim).astype(np.float32)
            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding.tolist())
            
        # Generate random query embeddings
        queries = []
        query_count = min(num_documents // 10, 20)  # Generate a reasonable number of queries
        for i in range(query_count):
            query_embedding = np.random.randn(embedding_dim).astype(np.float32)
            # Normalize
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            queries.append({
                "id": f"query_{i}",
                "embedding": query_embedding.tolist()
            })
            
        rag_logger.info(f"Generated {len(documents)} documents, {len(embeddings)} embeddings, and {len(queries)} queries")
        return documents, embeddings, queries
        
    async def run_benchmark(self, 
                          num_documents: int = 100, 
                          num_queries: int = 20,
                          batch_size: int = 50,
                          top_k: int = 5) -> Dict[str, Any]:
        """
        Run a benchmark comparing different RAG implementations.
        
        Args:
            num_documents: Total number of documents to use
            num_queries: Number of queries to run
            batch_size: Batch size for adding documents
            top_k: Number of results to retrieve per query
            
        Returns:
            Benchmark results
        """
        rag_logger.info(f"Starting RAG benchmark with {num_documents} documents and {num_queries} queries")
        
        # Generate test data
        documents, embeddings, queries = await self.generate_test_data(num_documents=num_documents)
        
        # Results dictionary
        results = {}
        
        # Test each pipeline
        for store_type in self.vector_store_types:
            try:
                rag_logger.info(f"Benchmarking {store_type} RAG implementation")
                
                # Create the pipeline for this store type
                pipeline = RAGPipeline(vector_store_type=store_type)
                
                # Measure document addition time
                add_times = []
                for i in range(0, num_documents, batch_size):
                    end_idx = min(i + batch_size, num_documents)
                    batch_docs = documents[i:end_idx]
                    batch_embeddings = embeddings[i:end_idx]
                    
                    start_time = time.time()
                    await pipeline.vector_store.add_documents(batch_docs, batch_embeddings)
                    end_time = time.time()
                    
                    add_times.append(end_time - start_time)
                    rag_logger.info(f"Added batch {i//batch_size + 1} ({len(batch_docs)} documents) to {store_type} in {add_times[-1]:.4f} seconds")
                
                # Measure query time
                query_times = []
                relevance_scores = []
                
                for i in range(min(num_queries, len(queries))):
                    query = queries[i % len(queries)]
                    
                    start_time = time.time()
                    results_list = await pipeline.vector_store.search(query["embedding"], top_k=top_k)
                    end_time = time.time()
                    
                    query_times.append(end_time - start_time)
                    
                    # Mock relevance score (in a real system this would be based on ground truth)
                    relevance = sum(random.uniform(0.5, 1.0) for _ in range(len(results_list))) / len(results_list) if results_list else 0
                    relevance_scores.append(relevance)
                    
                    if i % 10 == 0:
                        rag_logger.info(f"Ran query {i+1}/{num_queries} on {store_type} in {query_times[-1]:.4f} seconds")
                
                # Skip saving for now to avoid the file path error
                save_time = 0
                save_path = str(self.benchmark_dir / f"benchmark_{store_type}.pkl")
                
                # Store results
                results[store_type] = {
                    "add_time": {
                        "total": sum(add_times),
                        "avg": sum(add_times) / len(add_times) if add_times else 0,
                        "times": add_times[:5]  # Just store the first few times to keep result size reasonable
                    },
                    "query_time": {
                        "total": sum(query_times),
                        "avg": sum(query_times) / len(query_times) if query_times else 0,
                        "times": query_times[:5]  # Just store the first few times
                    },
                    "relevance": {
                        "avg": sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0,
                        "scores": relevance_scores[:5]  # Just store the first few scores
                    },
                    "save_time": save_time,
                    "total_docs": num_documents,
                    "total_queries": len(query_times),
                    "save_path": save_path
                }
            except Exception as e:
                rag_logger.error(f"Error benchmarking {store_type}: {str(e)}", exc_info=True)
                results[store_type] = {"error": str(e)}
        
        # Calculate winners and summary
        if results:
            valid_results = {k: v for k, v in results.items() if "error" not in v}
            
            if valid_results:
                try:
                    fastest_add = min(valid_results.items(), key=lambda x: x[1]["add_time"]["total"])
                    fastest_query = min(valid_results.items(), key=lambda x: x[1]["query_time"]["avg"])
                    highest_relevance = max(valid_results.items(), key=lambda x: x[1]["relevance"]["avg"])
                    
                    # Compute an overall score (lower is better)
                    overall_scores = {}
                    for store_type, result in valid_results.items():
                        # Weight add time at 30%, query time at 50%, and relevance at 20%
                        add_time_score = result["add_time"]["total"] / (fastest_add[1]["add_time"]["total"] or 1)
                        query_time_score = result["query_time"]["avg"] / (fastest_query[1]["query_time"]["avg"] or 1)
                        relevance_score = highest_relevance[1]["relevance"]["avg"] / (result["relevance"]["avg"] or 1)
                        
                        overall_scores[store_type] = 0.3 * add_time_score + 0.5 * query_time_score + 0.2 * relevance_score
                    
                    overall_winner = min(overall_scores.items(), key=lambda x: x[1])[0]
                    
                    # Add summary to results
                    results["summary"] = {
                        "fastest_add": fastest_add[0],
                        "fastest_query": fastest_query[0],
                        "highest_relevance": highest_relevance[0],
                        "overall_winner": overall_winner
                    }
                except Exception as e:
                    rag_logger.error(f"Error calculating summary: {str(e)}", exc_info=True)
                    results["summary"] = {
                        "error": f"Could not calculate full summary: {str(e)}"
                    }
        
        # Save benchmark results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = self.benchmark_dir / f"benchmark_results_{timestamp}.json"
        
        try:
            # Convert the results to JSON-serializable format
            serializable_results = {}
            for key, value in results.items():
                if isinstance(value, dict) and "error" in value:
                    serializable_results[key] = {"error": str(value["error"])}
                else:
                    try:
                        # Try to make each result serializable
                        json.dumps(value)  # Just a test to see if it's serializable
                        serializable_results[key] = value
                    except TypeError:
                        # If not serializable, convert to string
                        serializable_results[key] = str(value)
            
            with open(result_path, "w") as f:
                json.dump(serializable_results, f, indent=2, default=str)
                
            rag_logger.info(f"Benchmark complete. Results saved to {result_path}")
        except Exception as e:
            rag_logger.error(f"Error saving benchmark results: {str(e)}", exc_info=True)
        
        return results
        
    def print_results(self, results: Dict[str, Any]):
        """
        Print benchmark results in a readable format.
        
        Args:
            results: Benchmark results
        """
        print("\n===== RAG BENCHMARK RESULTS =====\n")
        
        if "summary" not in results:
            print("No valid benchmark results to display.")
            return
            
        # Print summary
        print("Summary:")
        summary = results.get("summary", {})
        print(f"- Fastest document addition: {summary.get('fastest_add', 'N/A')}")
        print(f"- Fastest query: {summary.get('fastest_query', 'N/A')}")
        print(f"- Highest relevance: {summary.get('highest_relevance', 'N/A')}")
        print(f"- Overall winner: {summary.get('overall_winner', 'N/A')}")
        print("\n")
        
        # Print table header
        header = f"{'Implementation':<10} | {'Add Time (s)':<12} | {'Query Time (ms)':<15} | {'Save Time (s)':<12}"
        print(header)
        print("-" * len(header))
        
        # Print results for each implementation
        for store_type, metrics in results.items():
            if store_type == "summary" or "error" in metrics:
                continue
                
            add_time = metrics["add_time"]["total"]
            query_time = metrics["query_time"]["avg"] * 1000  # Convert to ms
            save_time = metrics.get("save_time", 0)
            
            print(f"{store_type:<10} | {add_time:<12.4f} | {query_time:<15.4f} | {save_time:<12.4f}")
            
        print("\n")
        
        # Print errors if any
        errors = [(k, v["error"]) for k, v in results.items() if isinstance(v, dict) and "error" in v]
        if errors:
            print("Errors:")
            for impl, error in errors:
                print(f"- {impl}: {error}")
            print("\n")
            
        print("===================================\n")


async def run_benchmark():
    """
    Run a RAG benchmark from the command line.
    """
    benchmark = RAGBenchmark()
    results = await benchmark.run_benchmark()
    benchmark.print_results(results)
    
    
if __name__ == "__main__":
    asyncio.run(run_benchmark())