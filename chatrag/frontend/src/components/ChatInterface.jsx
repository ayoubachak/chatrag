import React, { useState, useEffect, useRef } from 'react';
import MessageList from './MessageList';
import FileUpload from './FileUpload';
import ModelSelector from './ModelSelector';
import ToggleButton from './ToggleButton';
import { useWebSocket } from '../services/websocket';

const ChatInterface = ({ userId, onConnectionChange }) => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [files, setFiles] = useState([]);
  const [useRag, setUseRag] = useState(false);
  const [modelType, setModelType] = useState('local'); // 'local', 'huggingface', or 'lm_studio'
  
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  
  // WebSocket connection
  const { sendMessage, lastMessage, readyState } = useWebSocket(
    `ws://${window.location.hostname}:8000/ws/chat/${userId}`
  );
  
  // Update connection status
  useEffect(() => {
    onConnectionChange(readyState === WebSocket.OPEN);
  }, [readyState, onConnectionChange]);
  
  // Handle incoming WebSocket messages
  useEffect(() => {
    if (lastMessage) {
      try {
        console.log('Received WebSocket message:', lastMessage.data);
        const data = JSON.parse(lastMessage.data);
        console.log('Parsed WebSocket data:', data);
        
        switch (data.type) {
          case 'message':
            // Add new message to the chat
            console.log('Adding message to chat:', data);
            setMessages(prev => [...prev, {
              id: data.id,
              role: data.role,
              content: data.content,
              timestamp: data.timestamp,
              metadata: data.metadata || {}
            }]);
            setIsTyping(false);
            break;
            
          case 'typing':
            // Update typing indicator
            setIsTyping(data.status === 'started');
            break;
            
          case 'file_processed':
            // Update file status
            setFiles(prev => 
              prev.map(file => 
                file.id === data.file_id 
                  ? { ...file, status: data.status, chunks: data.chunks }
                  : file
              )
            );
            break;
            
          case 'error':
            // Handle error
            console.error('Error from server:', data.error);
            setMessages(prev => [...prev, {
              id: 'error-' + Date.now(),
              role: 'system',
              content: `Error: ${data.error}`,
              timestamp: data.timestamp,
              isError: true
            }]);
            setIsTyping(false);
            break;
            
          case 'warning':
            // Handle warning
            console.warn('Warning from server:', data.message);
            setMessages(prev => [...prev, {
              id: 'warning-' + Date.now(),
              role: 'system',
              content: `Warning: ${data.message}`,
              timestamp: data.timestamp,
              isWarning: true
            }]);
            break;
            
          case 'heartbeat':
            // Ignore heartbeat messages
            console.log('Received heartbeat');
            break;
            
          case 'connection_status':
            // Handle connection status updates
            console.log('Connection status:', data.status);
            break;
            
          default:
            // Log unknown message types
            console.log('Unknown message type:', data.type);
            break;
        }
      } catch (err) {
        console.error('Error parsing WebSocket message:', err, lastMessage.data);
      }
    }
  }, [lastMessage]);
  
  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);
  
  // Focus input on load
  useEffect(() => {
    inputRef.current?.focus();
  }, []);
  
  // Handle message submission
  const handleSubmit = (e) => {
    e.preventDefault();
    
    if (!input.trim()) return;
    
    // Add user message to the chat
    const userMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: input,
      timestamp: new Date().toISOString()
    };
    
    setMessages(prev => [...prev, userMessage]);
    
    // Show typing indicator immediately
    setIsTyping(true);
    
    // Send message to server
    const success = sendMessage(JSON.stringify({
      type: 'message',
      content: input,
      use_rag: useRag,
      model_type: modelType,
      context: messages.slice(-10).map(msg => ({ 
        role: msg.role, 
        content: msg.content 
      }))
    }));
    
    // If sending failed, show an error
    if (!success) {
      setMessages(prev => [...prev, {
        id: 'error-' + Date.now(),
        role: 'system',
        content: 'Error: Failed to send message. Please check your connection and try again.',
        timestamp: new Date().toISOString(),
        isError: true
      }]);
      setIsTyping(false);
    }
    
    // Clear input
    setInput('');
  };
  
  // Handle file upload
  const handleFileUpload = (newFile) => {
    setFiles(prev => [...prev, newFile]);
  };
  
  return (
    <div className="chat-container">
      <div className="chat-sidebar">
        <div className="sidebar-section">
          <h3>Model Options</h3>
          <ModelSelector 
            modelType={modelType} 
            onChange={setModelType} 
          />
          
          <div className="toggle-options">
            <ToggleButton 
              label="Use RAG"
              active={useRag}
              onChange={setUseRag}
              icon="📚"
            />
          </div>
        </div>
        
        <div className="sidebar-section">
          <h3>Files</h3>
          <FileUpload 
            userId={userId}
            onFileUpload={handleFileUpload}
          />
          
          <div className="file-list">
            {files.length > 0 ? (
              files.map(file => (
                <div key={file.id} className={`file-item file-${file.status}`}>
                  <span className="file-name">{file.name}</span>
                  <span className="file-status">
                    {file.status === 'processing' && '⏳'}
                    {file.status === 'success' && '✅'}
                    {file.status === 'error' && '❌'}
                  </span>
                  {file.chunks && (
                    <span className="file-chunks">
                      {file.chunks} chunks
                    </span>
                  )}
                </div>
              ))
            ) : (
              <p className="no-files">No files uploaded yet</p>
            )}
          </div>
        </div>
      </div>
      
      <div className="chat-main">
        <div className="messages-container">
          <MessageList 
            messages={messages} 
            isTyping={isTyping}
          />
          <div ref={messagesEndRef} />
        </div>
        
        <form className="chat-input-form" onSubmit={handleSubmit}>
          <input
            ref={inputRef}
            type="text"
            className="chat-input"
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder="Type your message here..."
            disabled={readyState !== WebSocket.OPEN}
          />
          <button 
            type="submit" 
            className="send-button"
            disabled={!input.trim() || readyState !== WebSocket.OPEN}
          >
            Send
          </button>
        </form>
      </div>
    </div>
  );
};

export default ChatInterface;