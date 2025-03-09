# ChatRAG: A Retrieval-Augmented Generation Chat Application

This application demonstrates different Retrieval-Augmented Generation (RAG) implementations for enhancing LLM responses with relevant context from documents.

## Features

- Chat interface with multiple language model options
- Document upload and processing for RAG
- Multiple RAG implementations:
  - Basic: Simple in-memory vector store
  - FAISS: Fast similarity search using Facebook AI Similarity Search
  - ChromaDB: Persistent vector database with advanced features
  - Hybrid: Combined approach using FAISS for speed and ChromaDB for persistence
- Benchmarking utility to compare RAG implementations

## RAG Implementations

### Basic Vector Store

The basic implementation uses a simple in-memory vector store with numpy for similarity search. It's suitable for small document collections and development purposes.

**Pros:**
- Simple implementation
- No external dependencies
- Easy to understand

**Cons:**
- Not optimized for large document collections
- Slower similarity search
- No persistence by default

### FAISS Vector Store

The FAISS implementation uses Facebook AI Similarity Search for efficient vector search. It's significantly faster than the basic implementation, especially for large document collections.

**Pros:**
- Very fast similarity search
- Optimized for large document collections
- Supports various indexing methods

**Cons:**
- Requires additional dependencies
- Less feature-rich than dedicated vector databases
- In-memory by default (though can be saved/loaded)

### ChromaDB Vector Store

The ChromaDB implementation uses ChromaDB, a dedicated vector database for document retrieval. It provides persistence and advanced features like filtering and metadata management.

**Pros:**
- Persistent storage
- Advanced filtering capabilities
- Collection management
- Metadata support

**Cons:**
- Slower than FAISS for pure vector search
- More complex setup
- Additional dependencies

### Hybrid Vector Store

The hybrid implementation combines FAISS for fast in-memory search with ChromaDB for persistence. It provides the best of both worlds - speed and persistence.

**Pros:**
- Fast similarity search with FAISS
- Persistent storage with ChromaDB
- Fallback mechanism if one system fails
- Best overall performance

**Cons:**
- Most complex implementation
- More dependencies
- Higher memory usage

## Benchmarking

The application includes a benchmarking utility to compare the performance of different RAG implementations. You can run the benchmark with:

```bash
python benchmark_rag.py --documents 1000 --queries 100
```

Options:
- `--implementations`: Comma-separated list of implementations to benchmark (basic,faiss,chroma,hybrid)
- `--documents`: Number of documents to use in the benchmark
- `--queries`: Number of queries to run
- `--batch-size`: Batch size for adding documents
- `--top-k`: Number of results to retrieve per query
- `--output-dir`: Directory to save benchmark results

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python run.py
   ```

## Usage

1. Start the application
2. Upload documents through the web interface
3. Choose a RAG implementation (basic, FAISS, ChromaDB, or hybrid)
4. Chat with the model, which will use the uploaded documents for context

## API Endpoints

- `/upload`: Upload a document for RAG processing
- `/switch_rag_type`: Switch between RAG implementations
- `/ws/chat/{user_id}`: WebSocket endpoint for chat

## Configuration

Configuration options are available in the `.env` file:

- `MODEL_TYPE`: Default language model to use
- `RAG_TYPE`: Default RAG implementation to use
- `EMBEDDING_MODEL`: Model to use for generating embeddings
