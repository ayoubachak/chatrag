import { useState, useEffect, useCallback } from 'react';

export const useWebSocket = (url) => {
  const [socket, setSocket] = useState(null);
  const [lastMessage, setLastMessage] = useState(null);
  const [readyState, setReadyState] = useState(WebSocket.CONNECTING);
  const [reconnectAttempt, setReconnectAttempt] = useState(0);
  const maxReconnectAttempts = 5;
  const reconnectDelay = 3000; // 3 seconds
  const pingInterval = 30000; // 30 seconds

  // Initialize WebSocket connection
  useEffect(() => {
    // Create a new WebSocket connection
    const ws = new WebSocket(url);
    setSocket(ws);

    // Set up ping interval to keep connection alive
    const pingTimer = setInterval(() => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        // Send a ping to keep the connection alive
        try {
          ws.send(JSON.stringify({ type: 'ping' }));
        } catch (e) {
          console.error('Error sending ping:', e);
        }
      }
    }, pingInterval);

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
      try {
        // Parse the message to check if it's a heartbeat
        const data = JSON.parse(message.data);
        if (data.type === 'heartbeat') {
          // Just log heartbeat, don't update lastMessage
          console.log('Received heartbeat:', data.timestamp);
          return;
        }
      } catch (e) {
        // If parsing fails, treat as a regular message
      }
      
      // Set as last message for other message types
      setLastMessage(message);
    };

    // Clean up on unmount
    return () => {
      clearInterval(pingTimer);
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