from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from typing import Dict, List, Optional, Any, AsyncGenerator, Tuple, Callable, Union
import json
import uuid
import asyncio
from datetime import datetime
import logging
from enum import Enum, auto
from contextlib import asynccontextmanager

from models.base import BaseLanguageModel
from models.huggingface import HuggingFaceModel
from models.local_model import LocalModel
from models.lm_studio import LMStudioModel
from models.utils import format_chat_history
from rag.pipeline import RAGPipeline
from .connection import connection_manager
from .logger import chat_logger

router = APIRouter()

# Define message states for tracking lifecycle
class MessageState(Enum):
    INITIATED = auto()
    PROCESSING = auto()
    GENERATING = auto()
    STREAMING = auto()
    COMPLETED = auto()
    FAILED = auto()


class ModelFactory:
    """Factory class for creating language models."""
    
    _model_map = {
        "huggingface": HuggingFaceModel,
        "local": LocalModel,
        "lm_studio": LMStudioModel,
    }
    
    @classmethod
    def create(cls, model_type: str) -> BaseLanguageModel:
        """
        Create a language model instance based on the model type.
        
        Args:
            model_type: Type of model to use
            
        Returns:
            A language model instance
        
        Raises:
            ValueError: If the model type is invalid
        """
        model_class = cls._model_map.get(model_type)
        if not model_class:
            chat_logger.warning(f"Unknown model type '{model_type}', falling back to local model")
            model_class = LocalModel
            
        return model_class()


class ChatMessage:
    """Class representing a chat message with its complete state."""
    
    def __init__(self, connection_id: str, data: Dict):
        """Initialize a new chat message."""
        self.connection_id = connection_id
        self.data = data
        self.message_id = f"assistant-{uuid.uuid4()}"
        self.sources = []
        self.rag_pipeline = None
        self.model = None
        self.full_response = ""
        self.state = MessageState.INITIATED
        self.processing_started = datetime.now()
        self.last_activity = datetime.now()
        self.error = None
        
        # Extract options from message data
        self.prompt = data.get("content", "")
        self.use_rag = data.get("use_rag", False)
        self.model_type = data.get("model_type", "local")
        self.system_prompt = data.get("system_prompt", "You are a helpful assistant.")
        self.temperature = data.get("temperature", 0.7)
        self.max_tokens = data.get("max_tokens", 1024)
        self.rag_type = data.get("rag_type", "basic")
        self.chunking_strategy = data.get("chunking_strategy", "basic")
        self.use_streaming = data.get("use_streaming", False)
        self.context = data.get("context", [])
    
    def update_state(self, new_state: MessageState):
        """Update the message state and last activity timestamp."""
        self.state = new_state
        self.last_activity = datetime.now()
        chat_logger.debug(
            f"Message {self.message_id} state changed to {new_state.name} "
            f"(elapsed: {(self.last_activity - self.processing_started).total_seconds():.2f}s)"
        )
    
    def is_connection_active(self) -> bool:
        """Check if the connection is still active."""
        return self.connection_id in connection_manager.active_connections


@asynccontextmanager
async def manage_typing_indicator(message: ChatMessage):
    """Context manager to handle typing indicator state."""
    try:
        await send_typing_indicator(message.connection_id, "started")
        yield
    finally:
        await send_typing_indicator(message.connection_id, "stopped")


@asynccontextmanager
async def track_message_processing(message: ChatMessage):
    """Context manager to track message processing state and handle errors."""
    try:
        message.update_state(MessageState.PROCESSING)
        yield
        if message.state != MessageState.FAILED:  # Only update if not already failed
            message.update_state(MessageState.COMPLETED)
    except Exception as e:
        message.error = str(e)
        message.update_state(MessageState.FAILED)
        chat_logger.error(
            f"Error processing message {message.message_id}: {str(e)}",
            exc_info=True
        )
        if message.is_connection_active():
            await send_error(
                message.connection_id,
                f"Error processing message: {str(e)}"
            )
        raise  # Re-raise to ensure proper cleanup


async def process_chat_message(connection_id: str, data: Dict):
    """
    Process a chat message and generate a response.
    
    Args:
        connection_id: The connection ID
        data: The message data
    """
    # Create chat message object to track state
    message = ChatMessage(connection_id, data)
    
    # Validate prompt
    if not message.prompt:
        chat_logger.warning(f"Empty prompt received from user {connection_id}")
        return
    
    chat_logger.info(
        f"Processing chat message from user {connection_id}: "
        f"'{message.prompt[:50]}...' (use_rag={message.use_rag}, "
        f"model={message.model_type}, rag_type={message.rag_type}, "
        f"chunking_strategy={message.chunking_strategy}, "
        f"streaming={message.use_streaming})"
    )
    
    # Use context managers for consistent state tracking and cleanup
    async with manage_typing_indicator(message):
        async with track_message_processing(message):
            # Initialize model
            message.model = await initialize_model(message)
            if not message.model:
                return  # Error already handled
            
            # Process the message based on RAG configuration
            if message.use_rag:
                await process_rag_message(message)
            else:
                await process_standard_message(message)


async def initialize_model(message: ChatMessage) -> Optional[BaseLanguageModel]:
    """Initialize the language model."""
    try:
        return ModelFactory.create(message.model_type)
    except Exception as e:
        await handle_error(
            message.connection_id,
            f"Error initializing {message.model_type} model: {str(e)}",
            e
        )
        message.update_state(MessageState.FAILED)
        return None


async def process_rag_message(message: ChatMessage):
    """Process a message that uses RAG."""
    # Initialize RAG pipeline
    pipeline_initialized = await initialize_rag_pipeline(message)
    if not pipeline_initialized:
        # Fall back to non-RAG if pipeline initialization fails
        chat_logger.info(f"Falling back to non-RAG processing for {message.message_id}")
        await process_standard_message(message)
        return
    
    # Check if documents exist
    has_documents = await check_rag_documents(message)
    if not has_documents:
        await send_warning(
            message.connection_id,
            "No documents available for RAG. Please upload documents first."
        )
        # Fall back to standard processing
        await process_standard_message(message)
        return
    
    # Process with RAG
    try:
        chat_logger.info(f"Querying RAG pipeline with prompt: '{message.prompt[:50]}...'")
        retrieved_docs = await message.rag_pipeline.query(message.prompt, top_k=5)
        chat_logger.info(f"Retrieved {len(retrieved_docs)} documents from {message.rag_type} RAG")
        
        # Extract content and log documents
        doc_contents = [doc["content"] for doc in retrieved_docs]
        log_retrieved_documents(retrieved_docs)
        
        # Store sources for UI
        message.sources = [
            {
                "content": doc["content"],
                "metadata": doc["metadata"],
                "score": doc["score"]
            }
            for doc in retrieved_docs
        ]
        
        # Generate response
        if message.use_streaming:
            await stream_rag_response(message, doc_contents)
        else:
            await send_complete_rag_response(message, doc_contents)
            
    except Exception as e:
        chat_logger.error(
            f"Error in RAG processing with {message.model_type} model: {str(e)}",
            exc_info=True
        )
        
        # Send warning to user
        if message.is_connection_active():
            await send_warning(
                message.connection_id,
                f"Error using RAG with {message.model_type} model. Falling back to standard response."
            )
        
        # Fall back to standard processing
        await process_standard_message(message)


def log_retrieved_documents(docs: List[Dict]):
    """Log information about retrieved documents."""
    for i, doc in enumerate(docs):
        chat_logger.debug(
            f"Retrieved document {i+1}: "
            f"Score {doc['score']:.4f}, Content: '{doc['content'][:50]}...'"
        )


async def process_standard_message(message: ChatMessage):
    """Process a message without using RAG."""
    if message.use_streaming:
        await stream_standard_response(message)
    else:
        await send_complete_standard_response(message)


async def initialize_rag_pipeline(message: ChatMessage) -> bool:
    """Initialize the RAG pipeline."""
    try:
        # Get existing session or create new one
        vector_store_path = connection_manager.get_user_rag_session(message.connection_id)
        chat_logger.info(f"RAG session path for {message.connection_id}: {vector_store_path}")
        
        if not vector_store_path:
            # Create a new RAG pipeline
            return await create_new_rag_pipeline(message)
        else:
            # Use existing RAG pipeline
            return await load_existing_rag_pipeline(message, vector_store_path)
            
    except Exception as e:
        await handle_error(
            message.connection_id,
            f"Failed to initialize RAG pipeline: {str(e)}",
            e
        )
        return False


async def create_new_rag_pipeline(message: ChatMessage) -> bool:
    """Create a new RAG pipeline."""
    from rag.pipeline import create_default_rag_pipeline
    
    try:
        # Extract user ID from connection ID
        user_id = message.connection_id.split(":")[0] if ":" in message.connection_id else message.connection_id
        
        chat_logger.info(f"Creating default RAG pipeline for user {user_id} with type {message.rag_type}")
        message.rag_pipeline = await create_default_rag_pipeline(
            user_id=user_id,
            rag_type=message.rag_type,
            chunking_strategy=message.chunking_strategy
        )
        
        # Store session info
        connection_manager.set_user_rag_session(
            message.connection_id, 
            message.rag_pipeline.vector_store_path,
            message.rag_type,
            message.chunking_strategy
        )
        
        chat_logger.info(
            f"Created default RAG pipeline for user {message.connection_id} "
            f"with type {message.rag_type} at path {message.rag_pipeline.vector_store_path}"
        )
        return True
        
    except Exception as e:
        await handle_error(
            message.connection_id,
            f"Error creating RAG pipeline: {str(e)}",
            e
        )
        return False


async def load_existing_rag_pipeline(message: ChatMessage, vector_store_path: str) -> bool:
    """Load an existing RAG pipeline."""
    from rag.pipeline import load_rag_pipeline
    
    try:
        # Get current RAG settings
        user_rag_type = connection_manager.get_user_rag_type(message.connection_id)
        user_chunking_strategy = connection_manager.get_user_chunking_strategy(message.connection_id)
        
        # Update settings if changed
        if message.rag_type != user_rag_type or message.chunking_strategy != user_chunking_strategy:
            chat_logger.info(
                f"Updating RAG settings for user {message.connection_id}: "
                f"type={message.rag_type}, chunking={message.chunking_strategy}"
            )
            connection_manager.set_user_rag_session(
                message.connection_id, 
                vector_store_path,
                message.rag_type,
                message.chunking_strategy
            )
        
        # Load pipeline
        chat_logger.info(f"Loading RAG pipeline from {vector_store_path} with type {message.rag_type}")
        message.rag_pipeline = await load_rag_pipeline(
            vector_store_path=vector_store_path,
            rag_type=message.rag_type,
            chunking_strategy=message.chunking_strategy
        )
        
        # Log pipeline information
        log_rag_pipeline_info(message.rag_pipeline, message.rag_type)
        chat_logger.info(f"Loaded RAG pipeline for user {message.connection_id} with type {message.rag_type}")
        return True
        
    except Exception as e:
        await handle_error(
            message.connection_id,
            f"Error loading RAG pipeline: {str(e)}",
            e
        )
        return False


def log_rag_pipeline_info(rag_pipeline, rag_type: str):
    """Log information about the loaded RAG pipeline."""
    try:
        if rag_type == "basic" and hasattr(rag_pipeline.vector_store, 'embeddings'):
            chat_logger.info(f"Loaded basic vector store with {len(rag_pipeline.vector_store.embeddings)} embeddings")
        elif rag_type == "faiss" and hasattr(rag_pipeline.vector_store, 'index') and rag_pipeline.vector_store.index is not None:
            chat_logger.info(f"Loaded FAISS vector store with {rag_pipeline.vector_store.index.ntotal} vectors")
        elif rag_type == "chroma" and hasattr(rag_pipeline.vector_store, 'collection'):
            try:
                count = rag_pipeline.vector_store.collection.count()
                chat_logger.info(f"Loaded ChromaDB vector store with {count} documents")
            except Exception as e:
                chat_logger.warning(f"Error checking ChromaDB document count: {str(e)}")
    except Exception as e:
        chat_logger.warning(f"Error logging RAG pipeline info: {str(e)}")


async def check_rag_documents(message: ChatMessage) -> bool:
    """Check if the RAG pipeline has documents."""
    try:
        has_documents = False
        
        if message.rag_type == "basic" and hasattr(message.rag_pipeline.vector_store, 'embeddings'):
            has_documents = len(message.rag_pipeline.vector_store.embeddings) > 0
            chat_logger.info(f"Basic vector store has {len(message.rag_pipeline.vector_store.embeddings)} embeddings")
        elif (message.rag_type == "faiss" and 
              hasattr(message.rag_pipeline.vector_store, 'index') and 
              message.rag_pipeline.vector_store.index is not None):
            has_documents = message.rag_pipeline.vector_store.index.ntotal > 0
            chat_logger.info(f"FAISS vector store has {message.rag_pipeline.vector_store.index.ntotal} vectors")
        elif message.rag_type == "chroma" and hasattr(message.rag_pipeline.vector_store, 'collection'):
            try:
                count = message.rag_pipeline.vector_store.collection.count()
                has_documents = count > 0
                chat_logger.info(f"ChromaDB vector store has {count} documents")
            except Exception as e:
                chat_logger.warning(f"Error checking ChromaDB document count: {str(e)}")
                has_documents = False
        
        return has_documents
    except Exception as e:
        chat_logger.warning(f"Error checking for RAG documents: {str(e)}")
        return False


async def stream_rag_response(message: ChatMessage, doc_contents: List[str]):
    """Stream a response using RAG."""
    message.update_state(MessageState.STREAMING)
    
    # Send initial message to establish it in the UI
    await send_message_start(message)
    
    try:
        # Set up the generator
        generator = message.model.generate_with_rag_stream(
            prompt=message.prompt,
            documents=doc_contents,
            system_prompt=message.system_prompt,
            temperature=message.temperature,
            max_tokens=message.max_tokens,
            context=message.context
        )
        
        # Stream response chunks
        message.full_response = await stream_response_chunks(message, generator)
        
    except Exception as e:
        chat_logger.error(f"Error during RAG streaming: {str(e)}", exc_info=True)
        
        # Fall back to non-streaming if no content generated yet
        if not message.full_response and message.is_connection_active():
            chat_logger.info(f"RAG streaming failed, falling back to non-streaming response")
            try:
                message.full_response = await message.model.generate_with_rag(
                    prompt=message.prompt,
                    documents=doc_contents,
                    system_prompt=message.system_prompt,
                    temperature=message.temperature,
                    max_tokens=message.max_tokens,
                    context=message.context
                )
            except Exception as fallback_error:
                chat_logger.error(f"Fallback response generation failed: {str(fallback_error)}", exc_info=True)
                if message.is_connection_active():
                    await send_error(
                        message.connection_id,
                        "Failed to generate response. Please try again."
                    )
                return
    
    # Send message completion if connection still active
    if message.is_connection_active():
        await send_message_complete(message)
        chat_logger.info(f"Completed streaming RAG response for user {message.connection_id}")


async def send_complete_rag_response(message: ChatMessage, doc_contents: List[str]):
    """Send a complete response with RAG."""
    chat_logger.info(f"Generating response with RAG using {message.model_type} model")
    
    try:
        message.update_state(MessageState.GENERATING)
        response = await message.model.generate_with_rag(
            prompt=message.prompt,
            documents=doc_contents,
            system_prompt=message.system_prompt,
            temperature=message.temperature,
            max_tokens=message.max_tokens,
            context=message.context
        )
        
        # Validate response
        if not response or not isinstance(response, str):
            raise ValueError(f"Invalid response from {message.model_type} model with RAG: {response}")
        
        message.full_response = response
        
        # Send complete response if connection still active
        if message.is_connection_active():
            await connection_manager.send_message(
                message.connection_id,
                {
                    "type": "message",
                    "content": message.full_response,
                    "role": "assistant",
                    "id": message.message_id,
                    "metadata": {"sources": message.sources},
                    "timestamp": datetime.now().isoformat()
                }
            )
            chat_logger.info(f"Sent complete RAG response for user {message.connection_id}")
        else:
            chat_logger.warning(
                f"Connection {message.connection_id} no longer active, "
                "response not sent to UI"
            )
    
    except Exception as e:
        await handle_error(
            message.connection_id,
            f"Error generating RAG response: {str(e)}",
            e
        )
        message.update_state(MessageState.FAILED)


async def stream_standard_response(message: ChatMessage):
    """Stream a response without using RAG."""
    chat_logger.info(f"Generating streaming response without RAG using {message.model_type} model")
    message.update_state(MessageState.STREAMING)
    
    # Send initial message to establish it in the UI
    await send_message_start(message, include_sources=False)
    
    try:
        # Set up the generator
        generator = message.model.generate_stream(
            prompt=message.prompt,
            system_prompt=message.system_prompt,
            temperature=message.temperature,
            max_tokens=message.max_tokens,
            context=message.context
        )
        
        # Stream response chunks
        message.full_response = await stream_response_chunks(message, generator)
        
    except Exception as e:
        chat_logger.error(f"Error during standard streaming: {str(e)}", exc_info=True)
        
        # Fall back to non-streaming if no content generated yet
        if not message.full_response and message.is_connection_active():
            chat_logger.info(f"Standard streaming failed, falling back to non-streaming response")
            try:
                message.full_response = await message.model.generate(
                    prompt=message.prompt,
                    system_prompt=message.system_prompt,
                    temperature=message.temperature,
                    max_tokens=message.max_tokens,
                    context=message.context
                )
            except Exception as fallback_error:
                chat_logger.error(f"Fallback response generation failed: {str(fallback_error)}", exc_info=True)
                if message.is_connection_active():
                    await send_error(
                        message.connection_id,
                        "Failed to generate response. Please try again."
                    )
                return
    
    # Send message completion if connection still active
    if message.is_connection_active():
        await send_message_complete(message, include_sources=False)
        chat_logger.info(f"Completed streaming standard response for user {message.connection_id}")
    else:
        chat_logger.warning(
            f"Connection {message.connection_id} no longer active, "
            "standard streaming response completion not sent to UI"
        )


async def send_complete_standard_response(message: ChatMessage):
    """Send a complete response without using RAG."""
    chat_logger.info(f"Generating response without RAG using {message.model_type} model")
    
    try:
        message.update_state(MessageState.GENERATING)
        response = await message.model.generate(
            prompt=message.prompt,
            system_prompt=message.system_prompt,
            temperature=message.temperature,
            max_tokens=message.max_tokens,
            context=message.context
        )
        
        message.full_response = response
        chat_logger.info(f"Generated response for user {message.connection_id}: '{response[:50]}...'")
        
        # Send complete response with resilience
        if message.is_connection_active():
            await send_message_with_retry(
                message.connection_id,
                {
                    "type": "message",
                    "content": message.full_response,
                    "role": "assistant",
                    "id": message.message_id,
                    "metadata": {"sources": message.sources} if message.sources else {},
                    "timestamp": datetime.now().isoformat()
                }
            )
        else:
            chat_logger.warning(
                f"Connection {message.connection_id} no longer active, "
                "complete standard response not sent to UI"
            )
    
    except Exception as e:
        await handle_error(
            message.connection_id,
            f"Error generating standard response: {str(e)}",
            e
        )
        message.update_state(MessageState.FAILED)


async def stream_response_chunks(message: ChatMessage, generator: AsyncGenerator) -> str:
    """
    Stream response chunks from a generator.
    
    Args:
        message: The chat message
        generator: The response chunk generator
        
    Returns:
        str: The complete assembled response
    """
    full_response = ""
    
    # Set up backoff parameters for chunk sending resilience
    max_retries = 3
    base_delay = 0.2  # seconds
    
    try:
        # Process chunks from the generator
        while True:
            try:
                # Get the next chunk
                chunk = await generator.__anext__()
                if not chunk:  # Skip empty chunks
                    continue
                    
                full_response += chunk
                
                # Send chunk with retries
                for retry in range(max_retries):
                    if not message.is_connection_active():
                        chat_logger.warning(
                            f"Connection {message.connection_id} no longer active, "
                            "stopping chunk streaming"
                        )
                        return full_response
                        
                    try:
                        await connection_manager.send_message(
                            message.connection_id,
                            {
                                "type": "message_chunk",
                                "id": message.message_id,
                                "chunk": chunk,
                                "timestamp": datetime.now().isoformat()
                            }
                        )
                        break  # Successful send
                    except Exception as e:
                        if retry == max_retries - 1:  # Last retry
                            chat_logger.error(
                                f"Failed to send chunk after {max_retries} attempts: {str(e)}",
                                exc_info=True
                            )
                        else:
                            # Exponential backoff
                            delay = base_delay * (2 ** retry)
                            chat_logger.warning(
                                f"Chunk send failed (attempt {retry+1}/{max_retries}), "
                                f"retrying in {delay:.2f}s: {str(e)}"
                            )
                            await asyncio.sleep(delay)
                            
            except StopAsyncIteration:
                # End of generator
                break
            except Exception as chunk_error:
                chat_logger.error(f"Error processing chunk: {str(chunk_error)}", exc_info=True)
                break
                
    except Exception as e:
        chat_logger.error(f"Error in stream processing: {str(e)}", exc_info=True)
    
    return full_response


async def send_message_start(message: ChatMessage, include_sources: bool = True):
    """Send a message_start event to initialize streaming in the UI."""
    if not message.is_connection_active():
        chat_logger.warning(
            f"Connection {message.connection_id} no longer active, "
            "message_start not sent to UI"
        )
        return
        
    try:
        payload = {
            "type": "message_start",
            "role": "assistant",
            "id": message.message_id,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add sources metadata if available and requested
        if include_sources and message.sources:
            payload["metadata"] = {"sources": message.sources}
            
        await connection_manager.send_message(message.connection_id, payload)
    except Exception as e:
        chat_logger.error(f"Error sending message_start: {str(e)}", exc_info=True)
        # This is a critical error as it prevents UI from displaying the message
        # Make sure to explicitly set the message state to failed
        message.update_state(MessageState.FAILED)


async def send_message_complete(message: ChatMessage, include_sources: bool = True):
    """Send a message_complete event to finalize streaming in the UI."""
    if not message.is_connection_active():
        chat_logger.warning(
            f"Connection {message.connection_id} no longer active, "
            "message_complete not sent to UI"
        )
        return
        
    try:
        payload = {
            "type": "message_complete",
            "id": message.message_id,
            "content": message.full_response,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add sources metadata if available and requested
        if include_sources and message.sources:
            payload["metadata"] = {"sources": message.sources}
            
        await send_message_with_retry(message.connection_id, payload, max_retries=3)
    except Exception as e:
        chat_logger.error(f"Error sending message_complete: {str(e)}", exc_info=True)
        # This is a critical error as it can leave the UI in a "still typing" state
        # Try a direct error message to the UI
        try:
            await send_error(
                message.connection_id,
                "Error completing message. Please refresh the page if the response appears incomplete."
            )
        except:
            pass  # Already logged the original error
        
        message.update_state(MessageState.FAILED)


async def send_message_with_retry(
    connection_id: str,
    payload: Dict,
    max_retries: int = 2,
    initial_delay: float = 0.5
):
    """Send a message with retry logic."""
    if connection_id not in connection_manager.active_connections:
        chat_logger.warning(
            f"Connection {connection_id} not active, message not sent: "
            f"{payload.get('type', 'unknown_type')}"
        )
        return False
        
    # Define exponential backoff parameters
    delay = initial_delay
    
    for attempt in range(max_retries + 1):  # +1 for the initial attempt
        try:
            chat_logger.debug(
                f"Sending {payload.get('type', 'message')} to user {connection_id} "
                f"(attempt {attempt+1}/{max_retries+1})"
            )
            await connection_manager.send_message(connection_id, payload)
            chat_logger.debug(f"Successfully sent message to user {connection_id}")
            return True
            
        except Exception as e:
            # Last attempt failed
            if attempt == max_retries:
                chat_logger.error(
                    f"Failed to send message to user {connection_id} after {max_retries+1} attempts: {str(e)}",
                    exc_info=True
                )
                return False
                
            # Log the error and retry
            chat_logger.warning(
                f"Failed to send message to user {connection_id} (attempt {attempt+1}): {str(e)}"
            )
            
            # Check if connection is still active before retrying
            if connection_id not in connection_manager.active_connections:
                chat_logger.warning(
                    f"Connection {connection_id} no longer active during retry, aborting"
                )
                return False
                
            # Wait with exponential backoff before retrying
            await asyncio.sleep(delay)
            delay *= 2  # Exponential backoff


async def send_typing_indicator(connection_id: str, status: str):
    """Send a typing indicator message."""
    if connection_id not in connection_manager.active_connections:
        return
        
    try:
        await connection_manager.send_message(
            connection_id,
            {
                "type": "typing",
                "status": status,
                "timestamp": datetime.now().isoformat()
            }
        )
    except Exception as e:
        chat_logger.warning(f"Error sending typing indicator: {str(e)}")
        # Non-critical error, we can continue without the typing indicator


async def send_warning(connection_id: str, message: str):
    """Send a warning message to the client."""
    if connection_id not in connection_manager.active_connections:
        return
        
    try:
        await connection_manager.send_message(
            connection_id,
            {
                "type": "warning",
                "message": message,
                "timestamp": datetime.now().isoformat()
            }
        )
    except Exception as e:
        chat_logger.warning(f"Error sending warning message: {str(e)}")
        # Non-critical error, we can continue without the warning


async def send_error(connection_id: str, error_message: str):
    """Send an error message to the client."""
    if connection_id not in connection_manager.active_connections:
        return
        
    try:
        await connection_manager.send_message(
            connection_id,
            {
                "type": "error",
                "error": error_message,
                "timestamp": datetime.now().isoformat()
            }
        )
    except Exception as e:
        chat_logger.error(f"Error sending error message: {str(e)}")
        # Meta-error (error sending error), just log it


async def handle_error(connection_id: str, error_message: str, exception: Exception = None):
    """
    Handle errors by logging and sending an error message to the client.
    
    Args:
        connection_id: The connection ID
        error_message: The error message to send
        exception: The exception that caused the error, if any
    """
    if exception:
        chat_logger.error(error_message, exc_info=True)
    else:
        chat_logger.error(error_message)
    
    if connection_id in connection_manager.active_connections:
        await send_error(connection_id, error_message)


# ------------------------ WebSocket Handler -----------------------

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
    
    # Initialize connection monitoring
    connection_monitor_task = None
    heartbeat_task = None
    
    try:
        # Accept the connection
        await connection_manager.connect(websocket, connection_id)
        chat_logger.info(f"User {user_id} connected to chat websocket with session {session}")
        
        # Start monitoring tasks
        heartbeat_task = asyncio.create_task(send_heartbeat(connection_id))
        connection_monitor_task = asyncio.create_task(monitor_connection_health(connection_id))
        
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
        
        # Message processing loop
        while True:
            # Wait for messages with a reasonable timeout
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=60.0)
            except asyncio.TimeoutError:
                # No message received in timeout period, send a ping to check connection
                if connection_id in connection_manager.active_connections:
                    try:
                        await connection_manager.send_message(
                            connection_id,
                            {
                                "type": "ping",
                                "timestamp": datetime.now().isoformat()
                            }
                        )
                    except:
                        # If ping fails, connection is probably dead
                        chat_logger.warning(f"Ping failed for {connection_id}, closing connection")
                        break
                continue
            
            try:
                # Parse the message
                message_data = json.loads(data)
                
                # Process the message based on type
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
                    
                # Add any other message types here
                    
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
    except Exception as e:
        chat_logger.error(f"Error in websocket chat: {str(e)}", exc_info=True)
    finally:
        # Clean up resources
        connection_manager.disconnect(connection_id)
        
        # Cancel monitoring tasks
        if heartbeat_task:
            heartbeat_task.cancel()
        if connection_monitor_task:
            connection_monitor_task.cancel()
            
        chat_logger.info(f"Cleaned up resources for {connection_id}")


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
                try:
                    await connection_manager.send_message(
                        connection_id,
                        {
                            "type": "heartbeat",
                            "timestamp": datetime.now().isoformat()
                        }
                    )
                except Exception as e:
                    chat_logger.warning(f"Failed to send heartbeat to {connection_id}: {str(e)}")
    except asyncio.CancelledError:
        # Task was cancelled, exit gracefully
        pass
    except Exception as e:
        chat_logger.error(f"Error in heartbeat task: {str(e)}", exc_info=True)


async def monitor_connection_health(connection_id: str):
    """
    Monitor connection health and cleanup if issues detected.
    
    Args:
        connection_id: The connection ID
    """
    try:
        consecutive_failures = 0
        max_failures = 3
        
        while True:
            await asyncio.sleep(30)  # Check every 30 seconds
            
            if connection_id not in connection_manager.active_connections:
                # Connection already closed
                break
                
            # Try to send a lightweight ping
            try:
                await connection_manager.send_message(
                    connection_id,
                    {
                        "type": "connection_check",
                        "timestamp": datetime.now().isoformat()
                    }
                )
                # Success, reset failure counter
                consecutive_failures = 0
            except Exception as e:
                consecutive_failures += 1
                chat_logger.warning(
                    f"Connection check failed for {connection_id} "
                    f"({consecutive_failures}/{max_failures}): {str(e)}"
                )
                
                if consecutive_failures >= max_failures:
                    chat_logger.error(f"Connection {connection_id} unhealthy, forcing cleanup")
                    connection_manager.disconnect(connection_id)
                    break
                    
    except asyncio.CancelledError:
        # Task was cancelled, exit gracefully
        pass
    except Exception as e:
        chat_logger.error(f"Error in connection monitor: {str(e)}", exc_info=True)