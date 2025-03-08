import uvicorn

if __name__ == "__main__":
    # Run the server using the module path
    uvicorn.run("chatrag.backend.app:app", host="0.0.0.0", port=8000, reload=True) 