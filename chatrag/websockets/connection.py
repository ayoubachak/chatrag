from typing import Dict, List, Set, Optional
from fastapi import WebSocket
import json
import asyncio

class ConnectionManager:
    """
    Manager for WebSocket connections.
    """
    
    def __init__(self):
        """
        Initialize the connection manager.
        """
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_rag_sessions: Dict[str, str] = {}  # user_id -> vector_store_path
        
    async def connect(self, websocket: WebSocket, user_id: str):
        """
        Connect a new WebSocket and store it with the user ID.
        
        Args:
            websocket: The WebSocket connection
            user_id: The user's ID
        """
        await websocket.accept()
        self.active_connections[user_id] = websocket
        
    def disconnect(self, user_id: str):
        """
        Disconnect a WebSocket by user ID.
        
        Args:
            user_id: The user's ID
        """
        if user_id in self.active_connections:
            del self.active_connections[user_id]
            
    async def send_message(self, user_id: str, message: Dict):
        """
        Send a message to a specific user.
        
        Args:
            user_id: The user's ID
            message: The message to send
        """
        if user_id in self.active_connections:
            await self.active_connections[user_id].send_json(message)
            
    async def broadcast(self, message: Dict):
        """
        Broadcast a message to all connected users.
        
        Args:
            message: The message to broadcast
        """
        for connection in self.active_connections.values():
            await connection.send_json(message)
            
    def set_user_rag_session(self, user_id: str, vector_store_path: str):
        """
        Set the RAG session for a user.
        
        Args:
            user_id: The user's ID
            vector_store_path: Path to the user's vector store
        """
        self.user_rag_sessions[user_id] = vector_store_path
        
    def get_user_rag_session(self, user_id: str) -> Optional[str]:
        """
        Get the RAG session for a user.
        
        Args:
            user_id: The user's ID
            
        Returns:
            Path to the user's vector store, if any
        """
        return self.user_rag_sessions.get(user_id)


# Create a global instance
connection_manager = ConnectionManager()