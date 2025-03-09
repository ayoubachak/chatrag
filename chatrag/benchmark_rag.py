#!/usr/bin/env python
import asyncio
import argparse
from rag.benchmark import RAGBenchmark

async def main():
    """
    Run the RAG benchmark with command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Benchmark different RAG implementations")
    
    parser.add_argument("--implementations", type=str, default="basic,faiss,chroma,hybrid",
                        help="Comma-separated list of implementations to benchmark (basic,faiss,chroma,hybrid)")
    parser.add_argument("--documents", type=int, default=1000,
                        help="Number of documents to use in the benchmark")
    parser.add_argument("--queries", type=int, default=100,
                        help="Number of queries to run")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Batch size for adding documents")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of results to retrieve per query")
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                        help="Directory to save benchmark results")
    
    args = parser.parse_args()
    
    # Parse implementations
    implementations = [impl.strip() for impl in args.implementations.split(",")]
    
    # Validate implementations
    valid_implementations = ["basic", "faiss", "chroma", "hybrid"]
    for impl in implementations:
        if impl not in valid_implementations:
            print(f"Warning: Invalid implementation '{impl}'. Valid options are: {', '.join(valid_implementations)}")
            implementations.remove(impl)
    
    if not implementations:
        print(f"No valid implementations specified. Using all: {', '.join(valid_implementations)}")
        implementations = valid_implementations
    
    # Run benchmark
    benchmark = RAGBenchmark(
        vector_store_types=implementations,
        benchmark_dir=args.output_dir
    )
    
    print(f"Running benchmark with {args.documents} documents and {args.queries} queries")
    print(f"Testing implementations: {', '.join(implementations)}")
    
    results = await benchmark.run_benchmark(
        num_documents=args.documents,
        num_queries=args.queries,
        batch_size=args.batch_size,
        top_k=args.top_k
    )
    
    benchmark.print_results(results)
    
    print(f"Benchmark results saved to {args.output_dir}")

if __name__ == "__main__":
    asyncio.run(main()) 