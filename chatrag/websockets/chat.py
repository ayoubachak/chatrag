from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from typing import Dict, List, Optional, Any
import json
import uuid
import asyncio
from datetime import datetime

from models.base import BaseLanguageModel
from models.huggingface import HuggingFaceModel
from models.local_model import LocalModel
from models.lm_studio import LMStudioModel
from rag.pipeline import RAGPipeline
from .connection import connection_manager
from rag.logger import chat_logger

router = APIRouter()

# Model factory function
def get_model(model_type: str) -> BaseLanguageModel:
    """
    Get the appropriate model based on the type.
    
    Args:
        model_type: Type of model to use
        
    Returns:
        A language model instance
    """
    if model_type == "huggingface":
        return HuggingFaceModel()
    elif model_type == "local":
        return LocalModel()
    elif model_type == "lm_studio":
        return LMStudioModel()
    else:
        # Default to local model
        return LocalModel()

@router.websocket("/ws/chat/{user_id}")
async def websocket_chat(websocket: WebSocket, user_id: str):
    """
    WebSocket endpoint for chat.
    
    Args:
        websocket: The WebSocket connection
        user_id: The user's ID
    """
    await connection_manager.connect(websocket, user_id)
    chat_logger.info(f"User {user_id} connected to chat websocket")
    
    # Initialize or load RAG pipeline for this user
    rag_pipeline = RAGPipeline()
    rag_session = connection_manager.get_user_rag_session(user_id)
    if rag_session:
        try:
            chat_logger.info(f"Loading RAG session for user {user_id} from path: {rag_session}")
            rag_pipeline = await RAGPipeline.load(rag_session)
            chat_logger.info(f"RAG session loaded successfully. Vector store has {len(rag_pipeline.vector_store.embeddings)} embeddings")
        except Exception as e:
            # If loading fails, use a new pipeline
            chat_logger.error(f"Error loading RAG session: {str(e)}", exc_info=True)
            chat_logger.info(f"Using a new RAG pipeline instead")
    else:
        chat_logger.info(f"No existing RAG session found for user {user_id}")
    
    try:
        # Send initial connection acknowledgment
        await connection_manager.send_message(
            user_id, 
            {
                "type": "connection_status",
                "status": "connected",
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Chat loop
        while True:
            # Wait for messages
            data = await websocket.receive_text()
            
            try:
                # Parse the message
                message_data = json.loads(data)
                
                # Process the message
                if message_data.get("type") == "message":
                    chat_logger.info(f"Received chat message from user {user_id}")
                    await process_chat_message(user_id, message_data, rag_pipeline)
                    
                    # Save the RAG session after processing
                    try:
                        vector_store_path = await rag_pipeline.save()
                        connection_manager.set_user_rag_session(user_id, vector_store_path)
                        chat_logger.info(f"Saved RAG session after chat to {vector_store_path} with {len(rag_pipeline.vector_store.embeddings)} embeddings")
                    except Exception as e:
                        chat_logger.error(f"Error saving RAG session after chat: {str(e)}", exc_info=True)
                        
            except json.JSONDecodeError:
                chat_logger.error(f"Invalid JSON received from user {user_id}: {data}")
                await connection_manager.send_message(
                    user_id,
                    {
                        "type": "error",
                        "error": "Invalid message format",
                        "timestamp": datetime.now().isoformat()
                    }
                )
                
    except WebSocketDisconnect:
        chat_logger.info(f"User {user_id} disconnected from chat websocket")
        connection_manager.disconnect(user_id)
    except Exception as e:
        chat_logger.error(f"Error in websocket chat: {str(e)}", exc_info=True)
        try:
            # Try to save the RAG session before disconnecting
            vector_store_path = await rag_pipeline.save()
            connection_manager.set_user_rag_session(user_id, vector_store_path)
            chat_logger.info(f"Saved RAG session before disconnect to {vector_store_path}")
        except Exception as e:
            chat_logger.error(f"Error saving RAG session: {str(e)}", exc_info=True)

# In websockets/chat.py, modify the process_chat_message function:

async def process_chat_message(user_id: str, data: Dict, rag_pipeline: RAGPipeline):
    """
    Process a chat message and generate a response.
    
    Args:
        user_id: The user's ID
        data: The message data
        rag_pipeline: The RAG pipeline for this user
    """
    # Extract message content
    prompt = data.get("content", "")
    if not prompt:
        chat_logger.warning(f"Empty prompt received from user {user_id}")
        return
        
    # Extract options
    use_rag = data.get("use_rag", False)
    model_type = data.get("model_type", "local")
    system_prompt = data.get("system_prompt", "You are a helpful assistant.")
    temperature = data.get("temperature", 0.7)
    max_tokens = data.get("max_tokens", 1024)
    
    chat_logger.info(f"Processing chat message from user {user_id}: '{prompt[:50]}...' (use_rag={use_rag}, model={model_type})")
    
    # IMPORTANT CHANGE: Check if there's an updated RAG session and reload if necessary
    if use_rag:
        rag_session = connection_manager.get_user_rag_session(user_id)
        if rag_session:
            try:
                chat_logger.info(f"Reloading latest RAG session before processing: {rag_session}")
                rag_pipeline = await RAGPipeline.load(rag_session)
                chat_logger.info(f"Reloaded RAG pipeline with {len(rag_pipeline.vector_store.embeddings)} embeddings")
            except Exception as e:
                chat_logger.error(f"Error reloading RAG session: {str(e)}", exc_info=True)
    
    # Get conversation history (if provided)
    context = data.get("context", [])
    
    # Get the appropriate model
    model = get_model(model_type)
    
    # Send typing indicator
    await connection_manager.send_message(
        user_id,
        {
            "type": "typing",
            "status": "started",
            "timestamp": datetime.now().isoformat()
        }
    )
    
    try:
        # Generate response
        sources = []
        if use_rag:
            # Query the RAG system
            chat_logger.info(f"Using RAG for query: '{prompt[:50]}...' with vector store containing {len(rag_pipeline.vector_store.embeddings)} embeddings")
            
            # Skip RAG if no embeddings are available
            if not hasattr(rag_pipeline.vector_store, 'embeddings') or len(rag_pipeline.vector_store.embeddings) == 0:
                chat_logger.warning(f"No embeddings available in the vector store, falling back to standard response")
                await connection_manager.send_message(
                    user_id,
                    {
                        "type": "warning",
                        "message": "No document embeddings found. Using standard response instead.",
                        "timestamp": datetime.now().isoformat()
                    }
                )
                response = await model.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    context=context
                )
            else:
                retrieved_docs = await rag_pipeline.query(prompt, top_k=5)
                chat_logger.info(f"Retrieved {len(retrieved_docs)} documents from RAG")
                
                # Extract the content from retrieved documents
                doc_contents = [doc["content"] for doc in retrieved_docs]
                
                # Log retrieved documents
                for i, doc in enumerate(retrieved_docs):
                    chat_logger.debug(f"Retrieved document {i+1}: Score {doc['score']:.4f}, Content: '{doc['content'][:50]}...'")
                
                # Generate response with RAG context
                chat_logger.info(f"Generating response with RAG using {model_type} model")
                response = await model.generate_with_rag(
                    prompt=prompt,
                    documents=doc_contents,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    context=context
                )
                
                # Include retrieved documents in response metadata
                sources = [
                    {
                        "content": doc["content"],
                        "metadata": doc["metadata"],
                        "score": doc["score"]
                    }
                    for doc in retrieved_docs
                ]
        else:
            # Generate response without RAG
            chat_logger.info(f"Generating response without RAG using {model_type} model")
            response = await model.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                context=context
            )
            
        chat_logger.info(f"Generated response for user {user_id}: '{response[:50]}...'")
            
        # Send the response
        await connection_manager.send_message(
            user_id,
            {
                "type": "message",
                "content": response,
                "sources": sources,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Save the RAG pipeline if necessary
        if use_rag:
            try:
                # Get the current RAG session path
                rag_session = connection_manager.get_user_rag_session(user_id)
                
                # If we have a path, save to it; otherwise, generate a new one
                vector_store_path = await rag_pipeline.save(rag_session)
                connection_manager.set_user_rag_session(user_id, vector_store_path)
                chat_logger.info(f"Saved RAG pipeline after chat to {vector_store_path} with {len(rag_pipeline.vector_store.embeddings)} embeddings")
            except Exception as e:
                chat_logger.error(f"Error saving RAG session after chat: {str(e)}", exc_info=True)
            
    except Exception as e:
        # Send error message
        error_message = f"Error generating response: {str(e)}"
        chat_logger.error(error_message, exc_info=True)
        await connection_manager.send_message(
            user_id,
            {
                "type": "error",
                "error": error_message,
                "timestamp": datetime.now().isoformat()
            }
        )
    finally:
        # Stop typing indicator
        await connection_manager.send_message(
            user_id,
            {
                "type": "typing",
                "status": "stopped",
                "timestamp": datetime.now().isoformat()
            }
        )