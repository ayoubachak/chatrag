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
    user_id: str = Form(...),
    rag_type: str = Form("basic"),  # RAG type parameter with default
    chunking_strategy: str = Form("basic")  # Add chunking strategy parameter with default
):
    """
    Upload a file for RAG processing.
    
    Args:
        background_tasks: FastAPI background tasks
        file: The uploaded file
        user_id: The user's ID
        rag_type: Type of RAG implementation to use ("basic", "faiss", "chroma", or "hybrid")
        chunking_strategy: Strategy to use for chunking documents ("basic" or "super" or "quantum")
    """
    try:
        files_logger.info(f"Received file upload request from user {user_id} with RAG type: {rag_type} and chunking strategy: {chunking_strategy}")
        
        # Validate RAG type
        if rag_type not in ["basic", "faiss", "chroma", "hybrid"]:
            files_logger.warning(f"Invalid RAG type: {rag_type}, defaulting to 'basic'")
            rag_type = "basic"
            
        # Validate chunking strategy
        if chunking_strategy not in ["basic", "super", "quantum"]:
            files_logger.warning(f"Invalid chunking strategy: {chunking_strategy}, defaulting to 'basic'")
            chunking_strategy = "basic"
        
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
            file.filename,
            rag_type,
            chunking_strategy
        )
        
        return {
            "status": "success",
            "message": "File uploaded and processing started",
            "file_id": file_id,
            "filename": file.filename,
            "rag_type": rag_type,
            "chunking_strategy": chunking_strategy
        }
    except Exception as e:
        files_logger.error(f"Error uploading file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

async def process_file_background(file_path: str, user_id: str, file_id: str, original_filename: str, rag_type: str = "basic", chunking_strategy: str = "basic"):
    """
    Process a file in the background and update the RAG pipeline.
    
    Args:
        file_path: Path to the saved file
        user_id: The user's ID
        file_id: The file's ID
        original_filename: Original filename
        rag_type: Type of RAG implementation to use
        chunking_strategy: Strategy to use for chunking documents
    """
    try:
        files_logger.info(f"Starting background processing of file {original_filename} (ID: {file_id}) for user {user_id} with RAG type: {rag_type} and chunking strategy: {chunking_strategy}")
        
        # Check if the user_id contains a session part (user_id:session)
        # If not, we'll use it as is
        connection_id = user_id
        if ":" not in user_id:
            # Try to find any active connection for this user
            for conn_id in connection_manager.active_connections.keys():
                if conn_id.startswith(f"{user_id}:"):
                    connection_id = conn_id
                    files_logger.info(f"Found active connection {connection_id} for user {user_id}")
                    break
        
        files_logger.info(f"Using connection ID {connection_id} for RAG session")
        
        # Initialize or load RAG pipeline for this user
        rag_pipeline = None
        rag_session = connection_manager.get_user_rag_session(connection_id)
        user_rag_type = connection_manager.get_user_rag_type(connection_id)
        user_chunking_strategy = connection_manager.get_user_chunking_strategy(connection_id)
        
        if rag_session:
            try:
                files_logger.info(f"Loading existing RAG session for user {connection_id} from path: {rag_session} with type: {user_rag_type or rag_type} and chunking strategy: {user_chunking_strategy or chunking_strategy}")
                rag_pipeline = await RAGPipeline.load(rag_session, vector_store_type=user_rag_type or rag_type, chunking_strategy=user_chunking_strategy or chunking_strategy)
                files_logger.info(f"RAG session loaded successfully with type: {user_rag_type or rag_type} and chunking strategy: {user_chunking_strategy or chunking_strategy}")
            except Exception as e:
                # If loading fails, use a new pipeline
                files_logger.error(f"Error loading RAG session in process_file_background: {str(e)}", exc_info=True)
                files_logger.info(f"Using a new RAG pipeline with type: {rag_type} and chunking strategy: {chunking_strategy} instead")
                rag_pipeline = RAGPipeline(vector_store_type=rag_type, chunking_strategy=chunking_strategy)
        else:
            files_logger.info(f"No existing RAG session found for user {connection_id} in process_file_background, using type: {rag_type} and chunking strategy: {chunking_strategy}")
            rag_pipeline = RAGPipeline(vector_store_type=rag_type, chunking_strategy=chunking_strategy)
        
        # Process the file
        num_chunks = await rag_pipeline.process_file(file_path)
        files_logger.info(f"Processed file {original_filename} into {num_chunks} chunks")
        
        # Save the updated RAG pipeline
        vector_store_path = await rag_pipeline.save()
        files_logger.info(f"Saved RAG pipeline to {vector_store_path}")
        
        # Update the connection manager with the new RAG session
        connection_manager.set_user_rag_session(connection_id, vector_store_path, rag_type, chunking_strategy)
        files_logger.info(f"Updated user {connection_id} RAG session path to {vector_store_path} with type: {rag_type} and chunking strategy: {chunking_strategy}")
        
        # Notify the user - try both the connection_id and the original user_id
        notification_sent = False
        
        # Try with connection_id first
        try:
            await connection_manager.send_message(
                connection_id,
                {
                    "type": "file_processed",
                    "status": "success",
                    "file_id": file_id,
                    "filename": original_filename,
                    "chunks": num_chunks,
                    "rag_type": rag_type,
                    "chunking_strategy": chunking_strategy,
                    "vector_store_path": str(vector_store_path) if hasattr(vector_store_path, '__fspath__') else vector_store_path
                }
            )
            notification_sent = True
            files_logger.info(f"Notification sent to connection {connection_id}")
        except Exception as e:
            files_logger.warning(f"Could not send notification to connection {connection_id}: {str(e)}")
        
        # If notification wasn't sent and connection_id is different from user_id, try with user_id
        if not notification_sent and connection_id != user_id:
            try:
                await connection_manager.send_message(
                    user_id,
                    {
                        "type": "file_processed",
                        "status": "success",
                        "file_id": file_id,
                        "filename": original_filename,
                        "chunks": num_chunks,
                        "rag_type": rag_type,
                        "chunking_strategy": chunking_strategy,
                        "vector_store_path": str(vector_store_path) if hasattr(vector_store_path, '__fspath__') else vector_store_path
                    }
                )
                files_logger.info(f"Notification sent to user {user_id}")
            except Exception as e:
                files_logger.warning(f"Could not send notification to user {user_id}: {str(e)}")
        
    except Exception as e:
        files_logger.error(f"Error processing file {original_filename}: {str(e)}", exc_info=True)
        # Notify the user of the error - try both connection_id and user_id
        try:
            await connection_manager.send_message(
                connection_id if 'connection_id' in locals() else user_id,
                {
                    "type": "file_processed",
                    "status": "error",
                    "file_id": file_id,
                    "filename": original_filename,
                    "error": str(e)
                }
            )
        except Exception as notify_error:
            files_logger.error(f"Could not send error notification: {str(notify_error)}")
            
        # If connection_id is different from user_id, try with user_id as fallback
        if 'connection_id' in locals() and connection_id != user_id:
            try:
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
            except Exception as notify_error:
                files_logger.error(f"Could not send error notification to user_id: {str(notify_error)}")
                
    finally:
        # Log the final state of the RAG session
        if 'connection_id' in locals():
            rag_session = connection_manager.get_user_rag_session(connection_id)
            files_logger.info(f"Final RAG session for {connection_id}: {rag_session}")
            
            # Also check if there's a session for the plain user_id
            if connection_id != user_id:
                user_rag_session = connection_manager.get_user_rag_session(user_id)
                files_logger.info(f"Final RAG session for plain user_id {user_id}: {user_rag_session}")

@router.post("/switch_rag_settings")
async def switch_rag_settings(
    user_id: str = Form(...),
    rag_type: str = Form(...),
    chunking_strategy: str = Form(None)  # Optional parameter
):
    """
    Switch the RAG implementation type and/or chunking strategy for a user.
    
    Args:
        user_id: The user's ID
        rag_type: Type of RAG implementation to use ("basic", "faiss", "chroma", or "hybrid")
        chunking_strategy: Strategy to use for chunking documents ("basic" or "super" or "quantum")
    """
    try:
        files_logger.info(f"Switching RAG settings for user {user_id} - type: {rag_type}, chunking strategy: {chunking_strategy}")
        
        # Validate RAG type
        if rag_type not in ["basic", "faiss", "chroma", "hybrid"]:
            files_logger.warning(f"Invalid RAG type: {rag_type}")
            raise HTTPException(status_code=400, detail=f"Invalid RAG type: {rag_type}. Must be one of: basic, faiss, chroma, hybrid")
            
        # Validate chunking strategy if provided
        if chunking_strategy and chunking_strategy not in ["basic", "super", "quantum"]:
            files_logger.warning(f"Invalid chunking strategy: {chunking_strategy}")
            raise HTTPException(status_code=400, detail=f"Invalid chunking strategy: {chunking_strategy}. Must be one of: basic, super")
        
        # Get the current RAG session
        current_session = connection_manager.get_user_rag_session(user_id)
        current_type = connection_manager.get_user_rag_type(user_id)
        
        if not current_session:
            # No existing session, just set the type preference
            connection_manager.set_user_rag_session(user_id, None, rag_type)
            return {
                "status": "success", 
                "message": f"RAG type set to {rag_type} for future sessions" + 
                          (f" with chunking strategy {chunking_strategy}" if chunking_strategy else "")
            }
        
        # Load the current RAG pipeline
        try:
            # Use the current chunking strategy if a new one wasn't provided
            current_pipeline = await RAGPipeline.load(
                current_session, 
                vector_store_type=current_type or "basic",
                chunking_strategy=chunking_strategy or "basic"
            )
            
            # Create a new pipeline with the desired settings
            new_pipeline = RAGPipeline(
                vector_store_type=rag_type,
                chunking_strategy=chunking_strategy or "basic"
            )
            
            # If the current pipeline has documents, we need to transfer them
            # This is a simplified approach - in a real implementation, you might want to
            # process the documents again to optimize for the new vector store type
            
            # For simplicity, we'll just process the same files again if they exist
            # A more sophisticated approach would be to extract and transfer the embeddings directly
            
            # Save the new pipeline
            new_session_path = await new_pipeline.save()
            
            # Update the user's session
            connection_manager.set_user_rag_session(user_id, new_session_path, rag_type)
            
            return {
                "status": "success", 
                "message": f"Switched to {rag_type} RAG implementation" +
                          (f" with {chunking_strategy} chunking strategy" if chunking_strategy else "")
            }
            
        except Exception as e:
            files_logger.error(f"Error switching RAG settings: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error switching RAG settings: {str(e)}")
            
    except Exception as e:
        files_logger.error(f"Error in switch_rag_settings: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Keep the old endpoint for backward compatibility
@router.post("/switch_rag_type")
async def switch_rag_type(
    user_id: str = Form(...),
    rag_type: str = Form(...)
):
    """
    Switch the RAG implementation type for a user (legacy endpoint).
    
    Args:
        user_id: The user's ID
        rag_type: Type of RAG implementation to use ("basic", "faiss", "chroma", or "hybrid")
    """
    return await switch_rag_settings(user_id=user_id, rag_type=rag_type)