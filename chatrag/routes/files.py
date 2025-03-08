from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional
import os
import uuid
from datetime import datetime
from pathlib import Path

from rag.pipeline import RAGPipeline
from websockets.connection import connection_manager
from rag.logger import files_logger

router = APIRouter()

@router.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: str = Form(...)
):
    """
    Upload a file for RAG processing.
    
    Args:
        background_tasks: FastAPI background tasks
        file: The uploaded file
        user_id: The user's ID
    """
    try:
        # Create uploads directory if it doesn't exist
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        # Generate a unique file ID
        file_id = str(uuid.uuid4())
        
        # Save the file
        file_path = upload_dir / f"{file_id}_{file.filename}"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
            
        files_logger.info(f"File uploaded: {file.filename} (ID: {file_id}) by user {user_id}")
            
        # Process the file in the background
        background_tasks.add_task(
            process_file_background,
            str(file_path),
            user_id,
            file_id,
            file.filename
        )
        
        return {
            "status": "success",
            "message": "File uploaded and processing started",
            "file_id": file_id,
            "filename": file.filename
        }
    except Exception as e:
        files_logger.error(f"Error uploading file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

async def process_file_background(file_path: str, user_id: str, file_id: str, original_filename: str):
    """
    Process a file in the background and update the RAG pipeline.
    
    Args:
        file_path: Path to the saved file
        user_id: The user's ID
        file_id: The file's ID
        original_filename: Original filename
    """
    try:
        files_logger.info(f"Starting background processing of file {original_filename} (ID: {file_id}) for user {user_id}")
        
        # Initialize or load RAG pipeline for this user
        rag_pipeline = RAGPipeline()
        rag_session = connection_manager.get_user_rag_session(user_id)
        if rag_session:
            try:
                files_logger.info(f"Loading existing RAG session for user {user_id} from path: {rag_session}")
                rag_pipeline = await RAGPipeline.load(rag_session)
                files_logger.info(f"RAG session loaded successfully. Vector store has {len(rag_pipeline.vector_store.embeddings)} embeddings")
            except Exception as e:
                # If loading fails, use a new pipeline
                files_logger.error(f"Error loading RAG session in process_file_background: {str(e)}", exc_info=True)
                files_logger.info(f"Using a new RAG pipeline instead")
        else:
            files_logger.info(f"No existing RAG session found for user {user_id} in process_file_background")
        
        # Process the file
        num_chunks = await rag_pipeline.process_file(file_path)
        
        # Save the updated RAG pipeline
        vector_store_path = await rag_pipeline.save()
        files_logger.info(f"Saved RAG pipeline to {vector_store_path} with {len(rag_pipeline.vector_store.embeddings)} embeddings")
        connection_manager.set_user_rag_session(user_id, vector_store_path)
        files_logger.info(f"Updated user {user_id} RAG session path to {vector_store_path}")
        
        # Notify the user
        await connection_manager.send_message(
            user_id,
            {
                "type": "file_processed",
                "status": "success",
                "file_id": file_id,
                "filename": original_filename,
                "chunks": num_chunks
            }
        )
    except Exception as e:
        files_logger.error(f"Error processing file {original_filename}: {str(e)}", exc_info=True)
        # Notify the user of the error
        await connection_manager.send_message(
            user_id,
            {
                "type": "file_processed",
                "status": "error",
                "file_id": file_id,
                "filename": original_filename,
                "error": str(e)
            }
        )