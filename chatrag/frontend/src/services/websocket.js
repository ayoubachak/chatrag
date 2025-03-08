import { useState, useEffect, useCallback } from 'react';

export const useWebSocket = (url) => {
  const [socket, setSocket] = useState(null);
  const [lastMessage, setLastMessage] = useState(null);
  const [readyState, setReadyState] = useState(WebSocket.CONNECTING);
  const [reconnectAttempt, setReconnectAttempt] = useState(0);
  const maxReconnectAttempts = 5;
  const reconnectDelay = 3000; // 3 seconds

  // Initialize WebSocket connection
  useEffect(() => {
    // Create a new WebSocket connection
    const ws = new WebSocket(url);
    setSocket(ws);

    // Connection event handlers
    ws.onopen = () => {
      console.log('WebSocket connected');
      setReadyState(WebSocket.OPEN);
      setReconnectAttempt(0);
    };

    ws.onclose = (event) => {
      console.log(`WebSocket closed: ${event.code} ${event.reason}`);
      setReadyState(WebSocket.CLOSED);
      
      // Attempt to reconnect if not intentionally closed
      if (!event.wasClean && reconnectAttempt < maxReconnectAttempts) {
        console.log(`Attempting to reconnect (${reconnectAttempt + 1}/${maxReconnectAttempts})...`);
        setTimeout(() => {
          setReconnectAttempt(prev => prev + 1);
        }, reconnectDelay);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setReadyState(WebSocket.CLOSED);
    };

    ws.onmessage = (message) => {
      setLastMessage(message);
    };

    // Clean up on unmount
    return () => {
      if (ws) {
        ws.close();
      }
    };
  }, [url, reconnectAttempt]);

  // Send a message
  const sendMessage = useCallback((message) => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(message);
      return true;
    }
    return false;
  }, [socket]);

  return {
    socket,
    lastMessage,
    readyState,
    sendMessage,
  };
};