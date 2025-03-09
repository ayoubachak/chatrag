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
    use_streaming = data.get("use_streaming", False)  # Get streaming preference
    
    chat_logger.info(f"Processing chat message from user {connection_id}: '{prompt[:50]}...' (use_rag={use_rag}, model={model_type}, rag_type={rag_type}, chunking_strategy={chunking_strategy}, streaming={use_streaming})")
    
    # Initialize RAG pipeline for this specific request if needed
    rag_pipeline = None
    
    if use_rag:
        # Get the user's RAG session path
        vector_store_path = connection_manager.get_user_rag_session(connection_id)
        chat_logger.info(f"RAG session path for {connection_id}: {vector_store_path}")
        
        if not vector_store_path:
            # No RAG session, create a default one
            from rag.pipeline import create_default_rag_pipeline
            
            # Create a default RAG pipeline
            try:
                # Extract the user ID from the connection ID (format: user_id:session)
                user_id = connection_id.split(":")[0] if ":" in connection_id else connection_id
                
                chat_logger.info(f"Creating default RAG pipeline for user {user_id} with type {rag_type}")
                rag_pipeline = await create_default_rag_pipeline(
                    user_id=user_id,
                    rag_type=rag_type,
                    chunking_strategy=chunking_strategy
                )
                
                # Store the RAG session path
                connection_manager.set_user_rag_session(
                    connection_id, 
                    rag_pipeline.vector_store_path,
                    rag_type,
                    chunking_strategy
                )
                
                chat_logger.info(f"Created default RAG pipeline for user {connection_id} with type {rag_type} at path {rag_pipeline.vector_store_path}")
            except Exception as e:
                error_message = f"Error creating RAG pipeline: {str(e)}"
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
        else:
            # Use existing RAG session
            from rag.pipeline import load_rag_pipeline
            
            # Get the RAG type and chunking strategy for this user
            user_rag_type = connection_manager.get_user_rag_type(connection_id)
            user_chunking_strategy = connection_manager.get_user_chunking_strategy(connection_id)
            
            # If the requested RAG type or chunking strategy is different, update it
            if rag_type != user_rag_type or chunking_strategy != user_chunking_strategy:
                chat_logger.info(f"Updating RAG settings for user {connection_id}: type={rag_type}, chunking={chunking_strategy}")
                connection_manager.set_user_rag_session(
                    connection_id, 
                    vector_store_path,
                    rag_type,
                    chunking_strategy
                )
            
            # Load the RAG pipeline
            try:
                chat_logger.info(f"Loading RAG pipeline from {vector_store_path} with type {rag_type}")
                rag_pipeline = await load_rag_pipeline(
                    vector_store_path=vector_store_path,
                    rag_type=rag_type,
                    chunking_strategy=chunking_strategy
                )
                
                # Debug information about the loaded pipeline
                if rag_type == "basic" and hasattr(rag_pipeline.vector_store, 'embeddings'):
                    chat_logger.info(f"Loaded basic vector store with {len(rag_pipeline.vector_store.embeddings)} embeddings")
                elif rag_type == "faiss" and hasattr(rag_pipeline.vector_store, 'index') and rag_pipeline.vector_store.index is not None:
                    chat_logger.info(f"Loaded FAISS vector store with {rag_pipeline.vector_store.index.ntotal} vectors")
                elif rag_type == "chroma" and hasattr(rag_pipeline.vector_store, 'collection'):
                    try:
                        count = rag_pipeline.vector_store.collection.count()
                        has_documents = count > 0
                        chat_logger.info(f"Loaded ChromaDB vector store with {count} documents")
                    except Exception as e:
                        chat_logger.warning(f"Error checking ChromaDB document count: {str(e)}")
                        has_documents = False
                
                chat_logger.info(f"Loaded RAG pipeline for user {connection_id} with type {rag_type}")
            except Exception as e:
                error_message = f"Error loading RAG pipeline: {str(e)}"
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
        # Get chat history from the request
        context = data.get("context", [])
        
        # Generate response
        sources = []
        message_id = f"assistant-{uuid.uuid4()}"
        
        if use_rag and rag_pipeline:
            # Query the RAG system
            chat_logger.info(f"Using RAG ({rag_type}) for query: '{prompt[:50]}...'")
            
            # Check if vector store has documents (different check for each type)
            has_documents = False
            if rag_type == "basic" and hasattr(rag_pipeline.vector_store, 'embeddings'):
                has_documents = len(rag_pipeline.vector_store.embeddings) > 0
                chat_logger.info(f"Basic vector store has {len(rag_pipeline.vector_store.embeddings)} embeddings")
            elif rag_type == "faiss" and hasattr(rag_pipeline.vector_store, 'index') and rag_pipeline.vector_store.index is not None:
                has_documents = rag_pipeline.vector_store.index.ntotal > 0
                chat_logger.info(f"FAISS vector store has {rag_pipeline.vector_store.index.ntotal} vectors")
            elif rag_type == "chroma" and hasattr(rag_pipeline.vector_store, 'collection'):
                try:
                    count = rag_pipeline.vector_store.collection.count()
                    has_documents = count > 0
                    chat_logger.info(f"ChromaDB vector store has {count} documents")
                except Exception as e:
                    chat_logger.warning(f"Error checking ChromaDB document count: {str(e)}")
                    has_documents = False
            
            # Skip RAG if no documents are available
            if not has_documents:
                chat_logger.warning(f"No documents in {rag_type} vector store, skipping RAG")
                await connection_manager.send_message(
                    connection_id,
                    {
                        "type": "warning",
                        "message": "No documents available for RAG. Please upload documents first.",
                        "timestamp": datetime.now().isoformat()
                    }
                )
                
                # Fall back to regular generation
                if use_streaming:
                    # Generate streaming response without RAG
                    chat_logger.info(f"Generating streaming response without RAG using {model_type} model")
                    
                    # Send initial message to establish the message in the UI
                    await connection_manager.send_message(
                        connection_id,
                        {
                            "type": "message_start",
                            "role": "assistant",
                            "id": message_id,
                            "timestamp": datetime.now().isoformat()
                        }
                    )
                    
                    # Stream the response
                    full_response = ""
                    try:
                        # Use async for to handle the async generator
                        generator = model.generate_stream(
                            prompt=prompt,
                            system_prompt=system_prompt,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            context=context
                        )
                        
                        # Manually iterate through the generator
                        while True:
                            try:
                                # Get the next chunk
                                chunk = await generator.__anext__()
                                if chunk:  # Only send non-empty chunks
                                    full_response += chunk
                                    await connection_manager.send_message(
                                        connection_id,
                                        {
                                            "type": "message_chunk",
                                            "id": message_id,
                                            "chunk": chunk,
                                            "timestamp": datetime.now().isoformat()
                                        }
                                    )
                            except StopAsyncIteration:
                                # End of generator
                                break
                            except Exception as chunk_error:
                                chat_logger.error(f"Error processing chunk: {str(chunk_error)}", exc_info=True)
                                break
                    except Exception as stream_error:
                        chat_logger.error(f"Error during streaming: {str(stream_error)}", exc_info=True)
                        # If streaming fails, send what we have so far
                        if not full_response:
                            # If we have nothing, generate a non-streaming response as fallback
                            chat_logger.info(f"Streaming failed, falling back to non-streaming response")
                            full_response = await model.generate(
                                prompt=prompt,
                                system_prompt=system_prompt,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                context=context
                            )
                    
                    # Send message completion
                    await connection_manager.send_message(
                        connection_id,
                        {
                            "type": "message_complete",
                            "id": message_id,
                            "content": full_response,
                            "timestamp": datetime.now().isoformat()
                        }
                    )
                    
                    chat_logger.info(f"Completed streaming response for user {connection_id}")
                else:
                    # Generate non-streaming response without RAG
                    chat_logger.info(f"Generating response without RAG using {model_type} model")
                    response = await model.generate(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        context=context
                    )
                    
                    # Send the complete response
                    await connection_manager.send_message(
                        connection_id,
                        {
                            "type": "message",
                            "content": response,
                            "role": "assistant",
                            "id": message_id,
                            "timestamp": datetime.now().isoformat()
                        }
                    )
            else:
                try:
                    chat_logger.info(f"Querying RAG pipeline with prompt: '{prompt[:50]}...'")
                    retrieved_docs = await rag_pipeline.query(prompt, top_k=5)
                    chat_logger.info(f"Retrieved {len(retrieved_docs)} documents from {rag_type} RAG")
                    
                    # Extract the content from retrieved documents
                    doc_contents = [doc["content"] for doc in retrieved_docs]
                    
                    # Log retrieved documents
                    for i, doc in enumerate(retrieved_docs):
                        chat_logger.debug(f"Retrieved document {i+1}: Score {doc['score']:.4f}, Content: '{doc['content'][:50]}...'")
                    
                    # Include retrieved documents in response metadata
                    sources = [
                        {
                            "content": doc["content"],
                            "metadata": doc["metadata"],
                            "score": doc["score"]
                        }
                        for doc in retrieved_docs
                    ]
                    
                    if use_streaming:
                        # Generate streaming response with RAG
                        chat_logger.info(f"Generating streaming response with RAG using {model_type} model")
                        
                        # Send initial message to establish the message in the UI
                        await connection_manager.send_message(
                            connection_id,
                            {
                                "type": "message_start",
                                "role": "assistant",
                                "id": message_id,
                                "metadata": {"sources": sources},
                                "timestamp": datetime.now().isoformat()
                            }
                        )
                        
                        # Stream the response
                        full_response = ""
                        try:
                            # Use async for to handle the async generator
                            generator = model.generate_with_rag_stream(
                                prompt=prompt,
                                documents=doc_contents,
                                system_prompt=system_prompt,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                context=context
                            )
                            
                            # Manually iterate through the generator
                            while True:
                                try:
                                    # Get the next chunk
                                    chunk = await generator.__anext__()
                                    if chunk:  # Only send non-empty chunks
                                        full_response += chunk
                                        await connection_manager.send_message(
                                            connection_id,
                                            {
                                                "type": "message_chunk",
                                                "id": message_id,
                                                "chunk": chunk,
                                                "timestamp": datetime.now().isoformat()
                                            }
                                        )
                                except StopAsyncIteration:
                                    # End of generator
                                    break
                                except Exception as chunk_error:
                                    chat_logger.error(f"Error processing RAG chunk: {str(chunk_error)}", exc_info=True)
                                    break
                        except Exception as stream_error:
                            chat_logger.error(f"Error during RAG streaming: {str(stream_error)}", exc_info=True)
                            # If streaming fails, send what we have so far
                            if not full_response:
                                # If we have nothing, generate a non-streaming response as fallback
                                chat_logger.info(f"RAG streaming failed, falling back to non-streaming response")
                                full_response = await model.generate_with_rag(
                                    prompt=prompt,
                                    documents=doc_contents,
                                    system_prompt=system_prompt,
                                    temperature=temperature,
                                    max_tokens=max_tokens,
                                    context=context
                                )
                        
                        # Send message completion
                        await connection_manager.send_message(
                            connection_id,
                            {
                                "type": "message_complete",
                                "id": message_id,
                                "content": full_response,
                                "metadata": {"sources": sources},
                                "timestamp": datetime.now().isoformat()
                            }
                        )
                        
                        chat_logger.info(f"Completed streaming RAG response for user {connection_id}")
                    else:
                        # Generate non-streaming response with RAG
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
                        
                        # Send the complete response
                        await connection_manager.send_message(
                            connection_id,
                            {
                                "type": "message",
                                "content": response,
                                "role": "assistant",
                                "id": message_id,
                                "metadata": {"sources": sources},
                                "timestamp": datetime.now().isoformat()
                            }
                        )
                except Exception as e:
                    chat_logger.error(f"Error in RAG processing with {model_type} model: {str(e)}", exc_info=True)
                    # Fallback to non-RAG response
                    if use_streaming:
                        # Generate streaming response without RAG as fallback
                        chat_logger.info(f"Falling back to streaming response without RAG")
                        
                        # Send initial message to establish the message in the UI
                        await connection_manager.send_message(
                            connection_id,
                            {
                                "type": "message_start",
                                "role": "assistant",
                                "id": message_id,
                                "timestamp": datetime.now().isoformat()
                            }
                        )
                        
                        # Stream the response
                        full_response = ""
                        try:
                            # Use async for to handle the async generator
                            generator = model.generate_stream(
                                prompt=prompt,
                                system_prompt=system_prompt,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                context=context
                            )
                            
                            # Manually iterate through the generator
                            while True:
                                try:
                                    # Get the next chunk
                                    chunk = await generator.__anext__()
                                    if chunk:  # Only send non-empty chunks
                                        full_response += chunk
                                        await connection_manager.send_message(
                                            connection_id,
                                            {
                                                "type": "message_chunk",
                                                "id": message_id,
                                                "chunk": chunk,
                                                "timestamp": datetime.now().isoformat()
                                            }
                                        )
                                except StopAsyncIteration:
                                    # End of generator
                                    break
                                except Exception as chunk_error:
                                    chat_logger.error(f"Error processing fallback chunk: {str(chunk_error)}", exc_info=True)
                                    break
                        except Exception as stream_error:
                            chat_logger.error(f"Error during fallback streaming: {str(stream_error)}", exc_info=True)
                            # If streaming fails, send what we have so far
                            if not full_response:
                                # If we have nothing, generate a non-streaming response as fallback
                                chat_logger.info(f"Fallback streaming failed, using non-streaming response")
                                full_response = await model.generate(
                                    prompt=prompt,
                                    system_prompt=system_prompt,
                                    temperature=temperature,
                                    max_tokens=max_tokens,
                                    context=context
                                )
                        
                        # Send message completion
                        await connection_manager.send_message(
                            connection_id,
                            {
                                "type": "message_complete",
                                "id": message_id,
                                "content": full_response,
                                "timestamp": datetime.now().isoformat()
                            }
                        )
                    else:
                        # Generate non-streaming response without RAG as fallback
                        response = await model.generate(
                            prompt=prompt,
                            system_prompt=system_prompt,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            context=context
                        )
                        
                        # Send the complete response
                        await connection_manager.send_message(
                            connection_id,
                            {
                                "type": "message",
                                "content": response,
                                "role": "assistant",
                                "id": message_id,
                                "timestamp": datetime.now().isoformat()
                            }
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
            if use_streaming:
                # Generate streaming response without RAG
                chat_logger.info(f"Generating streaming response without RAG using {model_type} model")
                
                # Send initial message to establish the message in the UI
                await connection_manager.send_message(
                    connection_id,
                    {
                        "type": "message_start",
                        "role": "assistant",
                        "id": message_id,
                        "timestamp": datetime.now().isoformat()
                    }
                )
                
                # Stream the response
                full_response = ""
                try:
                    # Use async for to handle the async generator
                    generator = model.generate_stream(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        context=context
                    )
                    
                    # Manually iterate through the generator
                    while True:
                        try:
                            # Get the next chunk
                            chunk = await generator.__anext__()
                            if chunk:  # Only send non-empty chunks
                                full_response += chunk
                                await connection_manager.send_message(
                                    connection_id,
                                    {
                                        "type": "message_chunk",
                                        "id": message_id,
                                        "chunk": chunk,
                                        "timestamp": datetime.now().isoformat()
                                    }
                                )
                        except StopAsyncIteration:
                            # End of generator
                            break
                        except Exception as chunk_error:
                            chat_logger.error(f"Error processing chunk: {str(chunk_error)}", exc_info=True)
                            break
                except Exception as stream_error:
                    chat_logger.error(f"Error during streaming: {str(stream_error)}", exc_info=True)
                    # If streaming fails, send what we have so far
                    if not full_response:
                        # If we have nothing, generate a non-streaming response as fallback
                        chat_logger.info(f"Streaming failed, falling back to non-streaming response")
                        full_response = await model.generate(
                            prompt=prompt,
                            system_prompt=system_prompt,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            context=context
                        )
                
                # Send message completion
                await connection_manager.send_message(
                    connection_id,
                    {
                        "type": "message_complete",
                        "id": message_id,
                        "content": full_response,
                        "timestamp": datetime.now().isoformat()
                    }
                )
                
                chat_logger.info(f"Completed streaming response for user {connection_id}")
            else:
                # Generate non-streaming response without RAG
                chat_logger.info(f"Generating response without RAG using {model_type} model")
                response = await model.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    context=context
                )
                
                chat_logger.info(f"Generated response for user {connection_id}: '{response[:50]}...'")
                
                # Send the complete response
                try:
                    chat_logger.info(f"Sending response to user {connection_id}")
                    await connection_manager.send_message(
                        connection_id,
                        {
                            "type": "message",
                            "content": response,
                            "role": "assistant",
                            "id": message_id,
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
                                "id": message_id,
                                "metadata": {"sources": sources} if sources else {},
                                "timestamp": datetime.now().isoformat()
                            }
                        )
                        chat_logger.info(f"Response sent successfully on retry to user {connection_id}")
                    except Exception as retry_error:
                        chat_logger.error(f"Failed to send response on retry to user {connection_id}: {str(retry_error)}", exc_info=True)
    except Exception as e:
        error_message = f"Error processing message: {str(e)}"
        chat_logger.error(error_message, exc_info=True)
        await connection_manager.send_message(
            connection_id,
            {
                "type": "error",
                "error": error_message,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Stop typing indicator
        await connection_manager.send_message(
            connection_id,
            {
                "type": "typing",
                "status": "stopped",
                "timestamp": datetime.now().isoformat()
            }
        )