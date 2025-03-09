from typing import Dict, List, Set, Optional
from fastapi import WebSocket
import json
import asyncio
from .logger import connection_logger

class ConnectionManager:
    """
    Manager for WebSocket connections.
    """
    
    def __init__(self):
        """
        Initialize the connection manager.
        """
        self.active_connections = {}
        self.user_rag_sessions = {}
        self.user_rag_types = {}  # Track the RAG type for each user
        self.user_chunking_strategies = {}  # Track the chunking strategy for each user
        connection_logger.info("ConnectionManager initialized")
        
    async def connect(self, websocket: WebSocket, user_id: str):
        """
        Connect a new WebSocket and store it with the user ID.
        
        Args:
            websocket: The WebSocket connection
            user_id: The user's ID
        """
        await websocket.accept()
        self.active_connections[user_id] = websocket
        connection_logger.info(f"User {user_id} connected, total active connections: {len(self.active_connections)}")
        
    def disconnect(self, user_id: str):
        """
        Disconnect a WebSocket by user ID.
        
        Args:
            user_id: The user's ID
        """
        if user_id in self.active_connections:
            del self.active_connections[user_id]
            connection_logger.info(f"User {user_id} disconnected, remaining active connections: {len(self.active_connections)}")
        else:
            connection_logger.warning(f"Attempted to disconnect non-existent user: {user_id}")
            
    async def send_message(self, user_id: str, message: Dict):
        """
        Send a message to a specific user.
        
        Args:
            user_id: The user's ID
            message: The message to send
        """
        if user_id in self.active_connections:
            try:
                websocket = self.active_connections[user_id]
                if websocket.client_state.CONNECTED:
                    await websocket.send_json(message)
                    connection_logger.debug(f"Message sent to user {user_id}: {message.get('type', 'unknown')}")
                    return True
                else:
                    # Connection is not in CONNECTED state
                    connection_logger.warning(f"Cannot send message to user {user_id}: WebSocket not in CONNECTED state (state: {websocket.client_state})")
                    return False
            except Exception as e:
                connection_logger.error(f"Error sending message to user {user_id}: {str(e)}", exc_info=True)
                return False
        else:
            connection_logger.warning(f"Cannot send message to user {user_id}: User not found in active connections")
            return False
            
    async def broadcast(self, message: Dict):
        """
        Broadcast a message to all connected users.
        
        Args:
            message: The message to broadcast
        """
        for connection in self.active_connections.values():
            await connection.send_json(message)
            
    def set_user_rag_session(self, user_id: str, vector_store_path: str, rag_type: str = "basic", chunking_strategy: str = "basic"):
        """
        Set the RAG session path and type for a user.
        
        Args:
            user_id: The user's ID
            vector_store_path: Path to the vector store
            rag_type: Type of RAG implementation
            chunking_strategy: Strategy for chunking documents
        """
        self.user_rag_sessions[user_id] = vector_store_path
        self.user_rag_types[user_id] = rag_type
        self.user_chunking_strategies[user_id] = chunking_strategy
        connection_logger.info(f"Set RAG session for user {user_id}: path={vector_store_path}, type={rag_type}, chunking={chunking_strategy}")
        
    def get_user_rag_session(self, user_id: str) -> Optional[str]:
        """
        Get the RAG session path for a user.
        
        Args:
            user_id: The user's ID
            
        Returns:
            Path to the vector store or None if not set
        """
        return self.user_rag_sessions.get(user_id)
        
    def get_user_rag_type(self, user_id: str) -> str:
        """
        Get the RAG type for a user.
        
        Args:
            user_id: The user's ID
            
        Returns:
            RAG type or "basic" if not set
        """
        return self.user_rag_types.get(user_id, "basic")
        
    def get_user_chunking_strategy(self, user_id: str) -> str:
        """
        Get the chunking strategy for a user.
        
        Args:
            user_id: The user's ID
            
        Returns:
            Chunking strategy or "basic" if not set
        """
        return self.user_chunking_strategies.get(user_id, "basic")


# Create a global instance
connection_manager = ConnectionManager()