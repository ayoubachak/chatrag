from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from typing import Dict, List, Optional, Any
import json
import uuid
import asyncio
from datetime import datetime
import logging

from models.base import BaseLanguageModel
from models.huggingface import HuggingFaceModel
from models.local_model import LocalModel
from models.lm_studio import LMStudioModel
from models.utils import format_chat_history
from rag.pipeline import RAGPipeline
from .connection import connection_manager
from .logger import chat_logger

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
async def websocket_chat(
    websocket: WebSocket, 
    user_id: str, 
    session: str = None
):
    """
    WebSocket endpoint for chat.
    
    Args:
        websocket: The WebSocket connection
        user_id: The user's ID
        session: Optional session ID for managing multiple chat sessions
    """
    # Extract session from query parameters if not provided directly
    if not session:
        query_params = dict(websocket.query_params)
        session = query_params.get("session", str(uuid.uuid4()))
    
    # Create a unique connection ID combining user_id and session
    connection_id = f"{user_id}:{session}"
    
    await connection_manager.connect(websocket, connection_id)
    chat_logger.info(f"User {user_id} connected to chat websocket with session {session}")
    
    # Start a heartbeat task to keep the connection alive
    heartbeat_task = asyncio.create_task(send_heartbeat(connection_id))
    
    try:
        # Send initial connection acknowledgment
        await connection_manager.send_message(
            connection_id, 
            {
                "type": "connection_status",
                "status": "connected",
                "session": session,
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
                    chat_logger.info(f"Received chat message from user {user_id} in session {session}")
                    # Process message in a separate task to avoid blocking
                    asyncio.create_task(process_chat_message(connection_id, message_data))
                elif message_data.get("type") == "ping":
                    # Respond to ping with a pong to keep the connection alive
                    chat_logger.debug(f"Received ping from user {user_id} in session {session}")
                    await connection_manager.send_message(
                        connection_id,
                        {
                            "type": "pong",
                            "timestamp": datetime.now().isoformat()
                        }
                    )
                    
            except json.JSONDecodeError:
                chat_logger.error(f"Invalid JSON received from user {user_id}: {data}")
                await connection_manager.send_message(
                    connection_id,
                    {
                        "type": "error",
                        "error": "Invalid message format",
                        "timestamp": datetime.now().isoformat()
                    }
                )
                
    except WebSocketDisconnect:
        chat_logger.info(f"User {user_id} disconnected from chat websocket")
        connection_manager.disconnect(connection_id)
        # Cancel the heartbeat task
        heartbeat_task.cancel()
    except Exception as e:
        chat_logger.error(f"Error in websocket chat: {str(e)}", exc_info=True)
        # Cancel the heartbeat task
        heartbeat_task.cancel()


async def send_heartbeat(connection_id: str):
    """
    Send periodic heartbeat messages to keep the connection alive.
    
    Args:
        connection_id: The connection ID
    """
    try:
        while True:
            # Send a heartbeat every 15 seconds
            await asyncio.sleep(15)
            if connection_id in connection_manager.active_connections:
                await connection_manager.send_message(
                    connection_id,
                    {
                        "type": "heartbeat",
                        "timestamp": datetime.now().isoformat()
                    }
                )
    except asyncio.CancelledError:
        # Task was cancelled, exit gracefully
        pass
    except Exception as e:
        chat_logger.error(f"Error in heartbeat task: {str(e)}", exc_info=True)


async def process_chat_message(connection_id: str, data: Dict):
    """
    Process a chat message and generate a response.
    
    Args:
        connection_id: The connection ID
        data: The message data
    """
    # Extract message content
    prompt = data.get("content", "")
    if not prompt:
        chat_logger.warning(f"Empty prompt received from user {connection_id}")
        return
        
    # Extract options
    use_rag = data.get("use_rag", False)
    model_type = data.get("model_type", "local")
    system_prompt = data.get("system_prompt", "You are a helpful assistant.")
    temperature = data.get("temperature", 0.7)
    max_tokens = data.get("max_tokens", 1024)
    rag_type = data.get("rag_type", "basic")  # Get RAG type from request or use default
    chunking_strategy = data.get("chunking_strategy", "basic")  # Get chunking strategy from request or use default
    
    chat_logger.info(f"Processing chat message from user {connection_id}: '{prompt[:50]}...' (use_rag={use_rag}, model={model_type}, rag_type={rag_type}, chunking_strategy={chunking_strategy})")
    
    # Initialize RAG pipeline for this specific request if needed
    rag_pipeline = None
    
    if use_rag:
        # Load the RAG pipeline from the user's session
        rag_session = connection_manager.get_user_rag_session(connection_id)
        user_rag_type = connection_manager.get_user_rag_type(connection_id)
        user_chunking_strategy = connection_manager.get_user_chunking_strategy(connection_id)
        
        # Use the user's stored settings if available, otherwise use the ones from the request
        rag_type = user_rag_type if user_rag_type else rag_type
        chunking_strategy = user_chunking_strategy if user_chunking_strategy else chunking_strategy
        
        if rag_session:
            try:
                chat_logger.info(f"Loading RAG session for user {connection_id} from path: {rag_session} with type: {rag_type} and chunking strategy: {chunking_strategy}")
                rag_pipeline = await RAGPipeline.load(rag_session, vector_store_type=rag_type, chunking_strategy=chunking_strategy)
                chat_logger.info(f"Loaded RAG pipeline with type: {rag_type} and chunking strategy: {chunking_strategy}")
            except Exception as e:
                chat_logger.error(f"Error loading RAG session: {str(e)}", exc_info=True)
                # Create a new pipeline if loading fails
                rag_pipeline = RAGPipeline(vector_store_type=rag_type, chunking_strategy=chunking_strategy)
        else:
            # Create a new pipeline if no session exists
            chat_logger.info(f"No RAG session found for user {connection_id}, creating new pipeline with type: {rag_type} and chunking strategy: {chunking_strategy}")
            rag_pipeline = RAGPipeline(vector_store_type=rag_type, chunking_strategy=chunking_strategy)
    
    # Get conversation history (if provided)
    raw_context = data.get("context", [])
    
    # Format the conversation history properly for the LLM
    context = format_chat_history(raw_context, system_prompt=None)
    
    # Get the appropriate model
    try:
        model = get_model(model_type)
    except Exception as e:
        error_message = f"Error initializing {model_type} model: {str(e)}"
        chat_logger.error(error_message, exc_info=True)
        await connection_manager.send_message(
            connection_id,
            {
                "type": "error",
                "error": error_message,
                "timestamp": datetime.now().isoformat()
            }
        )
        return
    
    # Send typing indicator
    await connection_manager.send_message(
        connection_id,
        {
            "type": "typing",
            "status": "started",
            "timestamp": datetime.now().isoformat()
        }
    )
    
    try:
        # Generate response
        sources = []
        if use_rag and rag_pipeline:
            # Query the RAG system
            chat_logger.info(f"Using RAG ({rag_type}) for query: '{prompt[:50]}...'")
            
            # Check if vector store has documents (different check for each type)
            has_documents = False
            if rag_type == "basic" and hasattr(rag_pipeline.vector_store, 'document_count'):
                has_documents = rag_pipeline.vector_store.document_count > 0
            elif rag_type == "faiss" and rag_pipeline.vector_store.index is not None:
                has_documents = rag_pipeline.vector_store.index.ntotal > 0
            elif rag_type == "chroma":
                try:
                    has_documents = rag_pipeline.vector_store.collection.count() > 0
                except:
                    has_documents = False
            elif rag_type == "hybrid":
                try:
                    has_documents = rag_pipeline.vector_store.document_count > 0
                except:
                    has_documents = False
            
            # Skip RAG if no documents are available
            if not has_documents:
                chat_logger.warning(f"No documents available in the {rag_type} vector store, falling back to standard response")
                await connection_manager.send_message(
                    connection_id,
                    {
                        "type": "warning",
                        "message": f"No document embeddings found in {rag_type} store. Using standard response instead.",
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
                try:
                    retrieved_docs = await rag_pipeline.query(prompt, top_k=5)
                    chat_logger.info(f"Retrieved {len(retrieved_docs)} documents from {rag_type} RAG")
                    
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
                    
                    # Check if response is valid
                    if not response or not isinstance(response, str):
                        raise ValueError(f"Invalid response from {model_type} model with RAG: {response}")
                    
                    # Include retrieved documents in response metadata
                    sources = [
                        {
                            "content": doc["content"],
                            "metadata": doc["metadata"],
                            "score": doc["score"]
                        }
                        for doc in retrieved_docs
                    ]
                except Exception as e:
                    chat_logger.error(f"Error in RAG processing with {model_type} model: {str(e)}", exc_info=True)
                    # Fallback to non-RAG response
                    response = await model.generate(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        context=context
                    )
                    await connection_manager.send_message(
                        connection_id,
                        {
                            "type": "warning",
                            "message": f"Error using RAG with {model_type} model. Falling back to standard response.",
                            "timestamp": datetime.now().isoformat()
                        }
                    )
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
            
        chat_logger.info(f"Generated response for user {connection_id}: '{response[:50]}...'")
            
        # Send the response
        try:
            chat_logger.info(f"Sending response to user {connection_id}")
            await connection_manager.send_message(
                connection_id,
                {
                    "type": "message",
                    "content": response,
                    "role": "assistant",
                    "id": f"assistant-{uuid.uuid4()}",
                    "metadata": {"sources": sources} if sources else {},
                    "timestamp": datetime.now().isoformat()
                }
            )
            chat_logger.info(f"Response sent successfully to user {connection_id}")
        except Exception as e:
            chat_logger.error(f"Failed to send response to user {connection_id}: {str(e)}", exc_info=True)
            # Try one more time after a short delay
            try:
                await asyncio.sleep(0.5)
                await connection_manager.send_message(
                    connection_id,
                    {
                        "type": "message",
                        "content": response,
                        "role": "assistant",
                        "id": f"assistant-{uuid.uuid4()}",
                        "metadata": {"sources": sources} if sources else {},
                        "timestamp": datetime.now().isoformat()
                    }
                )
                chat_logger.info(f"Response sent successfully on retry to user {connection_id}")
            except Exception as retry_error:
                chat_logger.error(f"Failed to send response on retry to user {connection_id}: {str(retry_error)}", exc_info=True)
        
        # Save the RAG pipeline if it was used
        if use_rag and rag_pipeline:
            try:
                # Save to the user's RAG session
                vector_store_path = await rag_pipeline.save()
                connection_manager.set_user_rag_session(connection_id, vector_store_path, rag_type, chunking_strategy)
                chat_logger.info(f"Saved RAG pipeline to {vector_store_path} with {rag_pipeline.vector_store.document_count} documents")
            except Exception as e:
                chat_logger.error(f"Error saving RAG session: {str(e)}", exc_info=True)
            
    except Exception as e:
        # Send error message
        error_message = f"Error generating response: {str(e)}"
        chat_logger.error(error_message, exc_info=True)
        await connection_manager.send_message(
            connection_id,
            {
                "type": "error",
                "error": error_message,
                "timestamp": datetime.now().isoformat()
            }
        )
    finally:
        # Stop typing indicator
        await connection_manager.send_message(
            connection_id,
            {
                "type": "typing",
                "status": "stopped",
                "timestamp": datetime.now().isoformat()
            }
        )