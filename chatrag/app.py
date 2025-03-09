from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from pathlib import Path

# Import routes
from routes import files, benchmark
from websockets import chat
from dotenv import load_dotenv

load_dotenv(override=True)


# Create FastAPI app
app = FastAPI(title="RAG Chat Application")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(files.router, prefix="/api", tags=["files"])
app.include_router(chat.router, tags=["chat"])
app.include_router(benchmark.router, prefix="/api", tags=["benchmark"])

# Create uploads directory if it doesn't exist
uploads_dir = Path("uploads")
uploads_dir.mkdir(exist_ok=True)

# Create vector store directory if it doesn't exist
vector_store_dir = Path("vector_store")
vector_store_dir.mkdir(exist_ok=True)

# Create directories for different vector store types
Path("chroma_store").mkdir(exist_ok=True)
Path("hybrid_store").mkdir(exist_ok=True)
Path("benchmark_results").mkdir(exist_ok=True)

@app.get("/api/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "healthy"}

# Mount static files (frontend build)
# This allows serving the frontend from the same server
try:
    static_files_dir = Path("../frontend/dist")
    if static_files_dir.exists():
        app.mount("/", StaticFiles(directory=str(static_files_dir), html=True), name="static")
except Exception as e:
    print(f"Warning: Could not mount static files: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 8000))
    
    # Run the server
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)