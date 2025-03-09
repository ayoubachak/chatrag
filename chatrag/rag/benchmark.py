import time
import asyncio
import random
import string
import numpy as np
from typing import List, Dict, Any, Literal
from pathlib import Path
import os
import json

from .pipeline import RAGPipeline
from .logger import rag_logger

class RAGBenchmark:
    """
    Benchmark utility to compare the performance of different RAG implementations.
    """
    
    def __init__(self, 
                 vector_store_types: List[Literal["basic", "faiss", "chroma", "hybrid"]] = ["basic", "faiss", "chroma", "hybrid"],
                 benchmark_dir: str = "benchmark_results"):
        """
        Initialize the benchmark utility.
        
        Args:
            vector_store_types: List of vector store types to benchmark
            benchmark_dir: Directory to save benchmark results
        """
        self.vector_store_types = vector_store_types
        self.benchmark_dir = Path(benchmark_dir)
        self.benchmark_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize pipelines
        self.pipelines = {}
        for store_type in vector_store_types:
            self.pipelines[store_type] = RAGPipeline(vector_store_type=store_type)
            
        rag_logger.info(f"Initialized RAG benchmark with store types: {', '.join(vector_store_types)}")
        
    async def generate_test_data(self, 
                               num_documents: int = 100, 
                               embedding_dim: int = 384,
                               doc_length: int = 200):
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
        for i in range(10):  # 10 test queries
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
                          num_documents: int = 1000, 
                          num_queries: int = 100,
                          batch_size: int = 100,
                          top_k: int = 5):
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
        results = {
            "parameters": {
                "num_documents": num_documents,
                "num_queries": num_queries,
                "batch_size": batch_size,
                "top_k": top_k
            },
            "results": {}
        }
        
        # Test each pipeline
        for store_type, pipeline in self.pipelines.items():
            rag_logger.info(f"Benchmarking {store_type} RAG implementation")
            
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
            for i in range(min(num_queries, len(queries))):
                query = queries[i % len(queries)]
                
                start_time = time.time()
                results_list = await pipeline.vector_store.search(query["embedding"], top_k=top_k)
                end_time = time.time()
                
                query_times.append(end_time - start_time)
                
                if i % 10 == 0:
                    rag_logger.info(f"Ran query {i+1}/{num_queries} on {store_type} in {query_times[-1]:.4f} seconds")
            
            # Measure save time
            start_time = time.time()
            save_path = await pipeline.save(f"benchmark_{store_type}.pkl")
            end_time = time.time()
            save_time = end_time - start_time
            
            rag_logger.info(f"Saved {store_type} RAG pipeline in {save_time:.4f} seconds")
            
            # Measure load time
            start_time = time.time()
            loaded_pipeline = await RAGPipeline.load(save_path, vector_store_type=store_type)
            end_time = time.time()
            load_time = end_time - start_time
            
            rag_logger.info(f"Loaded {store_type} RAG pipeline in {load_time:.4f} seconds")
            
            # Store results
            results["results"][store_type] = {
                "add_time": {
                    "total": sum(add_times),
                    "average_per_batch": sum(add_times) / len(add_times),
                    "average_per_document": sum(add_times) / num_documents
                },
                "query_time": {
                    "total": sum(query_times),
                    "average": sum(query_times) / len(query_times)
                },
                "save_time": save_time,
                "load_time": load_time
            }
            
        # Save benchmark results
        timestamp = int(time.time())
        result_path = self.benchmark_dir / f"benchmark_results_{timestamp}.json"
        with open(result_path, "w") as f:
            json.dump(results, f, indent=2)
            
        rag_logger.info(f"Benchmark complete. Results saved to {result_path}")
        return results
        
    def print_results(self, results: Dict[str, Any]):
        """
        Print benchmark results in a readable format.
        
        Args:
            results: Benchmark results
        """
        print("\n===== RAG BENCHMARK RESULTS =====\n")
        print(f"Documents: {results['parameters']['num_documents']}")
        print(f"Queries: {results['parameters']['num_queries']}")
        print(f"Batch size: {results['parameters']['batch_size']}")
        print(f"Top-k: {results['parameters']['top_k']}")
        print("\n")
        
        # Print table header
        header = f"{'Implementation':<10} | {'Add Time (s)':<12} | {'Query Time (ms)':<15} | {'Save Time (s)':<12} | {'Load Time (s)':<12}"
        print(header)
        print("-" * len(header))
        
        # Print results for each implementation
        for store_type, metrics in results["results"].items():
            add_time = metrics["add_time"]["total"]
            query_time = metrics["query_time"]["average"] * 1000  # Convert to ms
            save_time = metrics["save_time"]
            load_time = metrics["load_time"]
            
            print(f"{store_type:<10} | {add_time:<12.4f} | {query_time:<15.4f} | {save_time:<12.4f} | {load_time:<12.4f}")
            
        print("\n")
        
        # Print summary
        print("Summary:")
        fastest_add = min(results["results"].items(), key=lambda x: x[1]["add_time"]["total"])[0]
        fastest_query = min(results["results"].items(), key=lambda x: x[1]["query_time"]["average"])[0]
        fastest_save = min(results["results"].items(), key=lambda x: x[1]["save_time"])[0]
        fastest_load = min(results["results"].items(), key=lambda x: x[1]["load_time"])[0]
        
        print(f"- Fastest document addition: {fastest_add}")
        print(f"- Fastest query: {fastest_query}")
        print(f"- Fastest save: {fastest_save}")
        print(f"- Fastest load: {fastest_load}")
        
        # Overall recommendation
        print("\nRecommendation:")
        if fastest_query == "faiss":
            print("- For speed-critical applications with many queries: Use FAISS")
        elif fastest_query == "hybrid":
            print("- For balanced performance and persistence: Use Hybrid")
        elif fastest_query == "chroma":
            print("- For persistence and advanced filtering: Use ChromaDB")
        else:
            print("- For simple use cases: Use Basic")
            
        print("\n===================================\n")


async def run_benchmark():
    """
    Run a RAG benchmark from the command line.
    """
    benchmark = RAGBenchmark()
    results = await benchmark.run_benchmark()
    benchmark.print_results(results)
    
    
if __name__ == "__main__":
    asyncio.run(run_benchmark()) 