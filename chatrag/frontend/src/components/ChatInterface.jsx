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
  const [ragType, setRagType] = useState('basic'); // 'basic', 'faiss', 'chroma', or 'hybrid'
  const [chunkingStrategy, setChunkingStrategy] = useState('basic'); // 'basic' or 'super' or 'quantum
  const [sessionId, setSessionId] = useState(Date.now().toString()); // Add a session ID
  const [useStreaming, setUseStreaming] = useState(false); // Add streaming state, default to true
  
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  
  // WebSocket connection
  const { sendMessage, lastMessage, readyState, reconnect } = useWebSocket(
    `ws://${window.location.hostname}:8000/ws/chat/${userId}?session=${sessionId}`
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
            
          case 'message_start':
            // Initialize a streaming message
            console.log('Starting streaming message:', data);
            setMessages(prev => [...prev, {
              id: data.id,
              role: data.role,
              content: '',
              timestamp: data.timestamp,
              metadata: data.metadata || {},
              isStreaming: true
            }]);
            break;
            
          case 'message_chunk':
            // Append to a streaming message
            console.log('Received message chunk:', data);
            setMessages(prev => 
              prev.map(msg => 
                msg.id === data.id 
                  ? { ...msg, content: msg.content + data.chunk }
                  : msg
              )
            );
            break;
            
          case 'message_complete':
            // Finalize a streaming message
            console.log('Completed streaming message:', data);
            setMessages(prev => 
              prev.map(msg => 
                msg.id === data.id 
                  ? { 
                      ...msg, 
                      content: data.content, // Use the complete content from server
                      isStreaming: false,
                      metadata: data.metadata || msg.metadata
                    }
                  : msg
              )
            );
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
                  ? { 
                      ...file, 
                      status: data.status, 
                      chunks: data.chunks,
                      ragType: data.rag_type || file.ragType // Update RAG type if provided
                    }
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
  
  // Add welcome message on first load
  useEffect(() => {
    // Only add welcome message if there are no messages yet
    if (messages.length === 0) {
      setMessages([{
        id: `welcome-${Date.now()}`,
        role: 'system',
        content: 'Welcome! How can I help you today?',
        timestamp: new Date().toISOString(),
        isSystemMessage: true  // Mark as system message for UI purposes only
      }]);
    }
  }, []);
  
  // Add keyboard shortcut for new chat (Ctrl+N)
  useEffect(() => {
    const handleKeyDown = (e) => {
      // Check if Ctrl+N is pressed
      if ((e.ctrlKey || e.metaKey) && e.key === 'n') {
        e.preventDefault(); // Prevent browser's "New Window" action
        handleNewChat();
      }
    };
    
    // Add event listener
    window.addEventListener('keydown', handleKeyDown);
    
    // Clean up
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
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
    
    // Filter out system messages and get only the last 10 user/assistant messages
    const chatHistory = messages
      .filter(msg => msg.role === 'user' || msg.role === 'assistant')
      .slice(-10)
      .map(msg => ({ 
        role: msg.role, 
        content: msg.content 
      }));
    
    // Send message to server
    const success = sendMessage(JSON.stringify({
      type: 'message',
      content: input,
      use_rag: useRag,
      model_type: modelType,
      rag_type: ragType,
      chunking_strategy: chunkingStrategy,
      use_streaming: useStreaming, // Add streaming preference
      system_prompt: "You are a helpful assistant. Provide accurate and concise information.",
      context: chatHistory
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

  // Handle RAG type change
  const handleRagTypeChange = async (newRagType) => {
    setRagType(newRagType);
    
    // If there are files already uploaded, notify the server about the RAG type change
    if (files.length > 0 && useRag) {
      try {
        const response = await fetch('/api/switch_rag_settings', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
          },
          body: new URLSearchParams({
            user_id: userId,
            rag_type: newRagType,
            chunking_strategy: chunkingStrategy
          })
        });
        
        if (!response.ok) {
          throw new Error(`Failed to switch RAG settings: ${response.status}`);
        }
        
        const result = await response.json();
        console.log('RAG settings updated:', result);
        
        // Add a system message about the change
        setMessages(prev => [...prev, {
          id: 'system-' + Date.now(),
          role: 'system',
          content: `RAG type changed to ${newRagType}. ${result.message}`,
          timestamp: new Date().toISOString()
        }]);
      } catch (error) {
        console.error('Error switching RAG type:', error);
        setMessages(prev => [...prev, {
          id: 'error-' + Date.now(),
          role: 'system',
          content: `Error: ${error.message}`,
          timestamp: new Date().toISOString(),
          isError: true
        }]);
      }
    }
  };

  // Handle chunking strategy change
  const handleChunkingStrategyChange = async (newStrategy) => {
    setChunkingStrategy(newStrategy);
    
    // If there are files already uploaded, notify the server about the chunking strategy change
    if (files.length > 0 && useRag) {
      try {
        const response = await fetch('/api/switch_rag_settings', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
          },
          body: new URLSearchParams({
            user_id: userId,
            rag_type: ragType,
            chunking_strategy: newStrategy
          })
        });
        
        if (!response.ok) {
          throw new Error(`Failed to switch chunking strategy: ${response.status}`);
        }
        
        const result = await response.json();
        console.log('Chunking strategy updated:', result);
        
        // Add a system message about the change
        setMessages(prev => [...prev, {
          id: 'system-' + Date.now(),
          role: 'system',
          content: `Chunking strategy changed to ${newStrategy === 'basic' ? 'Basic Chunking' : 'Super Chunking'}. ${result.message}`,
          timestamp: new Date().toISOString()
        }]);
      } catch (error) {
        console.error('Error switching chunking strategy:', error);
        setMessages(prev => [...prev, {
          id: 'error-' + Date.now(),
          role: 'system',
          content: `Error: ${error.message}`,
          timestamp: new Date().toISOString(),
          isError: true
        }]);
      }
    }
  };

  // Handle starting a new chat
  const handleNewChat = () => {
    // Clear messages
    setMessages([]);
    
    // Clear input
    setInput('');
    
    // Reset typing indicator
    setIsTyping(false);
    
    // Generate a new session ID
    const newSessionId = Date.now().toString();
    setSessionId(newSessionId);
    
    // Reconnect WebSocket with new session ID
    reconnect(`ws://${window.location.hostname}:8000/ws/chat/${userId}?session=${newSessionId}`);
    
    // Focus the input field
    inputRef.current?.focus();
    
    // Add welcome message
    setTimeout(() => {
      setMessages([{
        id: `system-${Date.now()}`,
        role: 'system',
        content: 'New chat started. How can I help you today?',
        timestamp: new Date().toISOString(),
        isSystemMessage: true  // Mark as system message for UI purposes only
      }]);
    }, 100);  // Small delay to ensure WebSocket reconnection has started
    
    console.log('Started new chat session:', newSessionId);
  };
  
  return (
    <div className="chat-container">
      <div className="chat-sidebar">
        <div className="sidebar-section">
          <button 
            className="new-chat-button" 
            onClick={handleNewChat}
            title="Start a new chat session (Ctrl+N)"
          >
            <span className="new-chat-icon">+</span>
            New Chat
          </button>
        </div>
        
        <div className="sidebar-section rag-section">
          <div className="section-header">
            <h3>RAG Settings</h3>
            <ToggleButton 
              isActive={useRag} 
              onClick={() => setUseRag(!useRag)}
              activeLabel="On"
              inactiveLabel="Off"
            />
          </div>
          
          {useRag ? (
            <div className="rag-settings-panel">
              <div className="control-group">
                <label>Vector Store Type:</label>
                <div className="rag-type-buttons">
                  {[
                    { id: 'basic', name: 'Basic', description: 'Simple in-memory vector store' },
                    { id: 'faiss', name: 'FAISS', description: 'Fast similarity search' },
                    { id: 'chroma', name: 'ChromaDB', description: 'Persistent vector database' },
                    { id: 'hybrid', name: 'Hybrid', description: 'FAISS + ChromaDB for optimal performance' }
                  ].map(type => (
                    <button
                      key={type.id}
                      className={`rag-type-button ${ragType === type.id ? 'active' : ''}`}
                      onClick={() => handleRagTypeChange(type.id)}
                      title={type.description}
                    >
                      {type.name}
                    </button>
                  ))}
                </div>
                <div className="rag-description">
                  {[
                    { id: 'basic', name: 'Basic', description: 'Simple in-memory vector store' },
                    { id: 'faiss', name: 'FAISS', description: 'Fast similarity search' },
                    { id: 'chroma', name: 'ChromaDB', description: 'Persistent vector database' },
                    { id: 'hybrid', name: 'Hybrid', description: 'FAISS + ChromaDB for optimal performance' }
                  ].find(type => type.id === ragType)?.description}
                </div>
              </div>
              
              <div className="control-group">
                <label>Chunking Strategy:</label>
                <div className="chunking-buttons">
                  {[
                    { id: 'basic', name: 'Basic', description: 'Simple text splitting' },
                    { id: 'super', name: 'Super', description: 'Advanced semantic chunking' },
                    { id: 'quantum', name: 'Quantum', description: 'Super Advanced semantic chunking' }
                  ].map(strategy => (
                    <button
                      key={strategy.id}
                      className={`chunking-button ${chunkingStrategy === strategy.id ? 'active' : ''}`}
                      onClick={() => handleChunkingStrategyChange(strategy.id)}
                      title={strategy.description}
                    >
                      {strategy.name}
                    </button>
                  ))}
                </div>
                <div className="chunking-description">
                  {[
                    { id: 'basic', name: 'Basic', description: 'Simple text splitting' },
                    { id: 'super', name: 'Super', description: 'Advanced semantic chunking' },
                    { id: 'quantum', name: 'Quantum', description: 'Super Advanced semantic chunking' }
                  ].find(strategy => strategy.id === chunkingStrategy)?.description}
                </div>
              </div>
            </div>
          ) : (
            <div className="rag-disabled-message">
              <p>RAG is currently disabled. Enable it to use document retrieval capabilities.</p>
            </div>
          )}
        </div>
        
        <div className="sidebar-section">
          <div className="section-header">
            <h3>Response Mode</h3>
            <ToggleButton 
              isActive={useStreaming} 
              onClick={() => setUseStreaming(!useStreaming)}
              activeLabel="Stream"
              inactiveLabel="Complete"
            />
          </div>
          <div className="setting-description">
            {useStreaming 
              ? "Responses will stream in real-time as they're generated" 
              : "Responses will be sent after complete generation"}
          </div>
        </div>
        
        <div className="sidebar-section">
          <h3>Model Options</h3>
          <ModelSelector 
            modelType={modelType} 
            onChange={setModelType} 
          />
        </div>
        
        <div className="sidebar-section">
          <h3>Upload Files</h3>
          <FileUpload 
            userId={userId} 
            onFileUpload={handleFileUpload} 
            ragType={ragType}
            chunkingStrategy={chunkingStrategy}
            useRag={useRag}
          />
          
          {files.length > 0 && (
            <div className="files-list">
              <h4>Uploaded Files</h4>
              <ul>
                {files.map(file => (
                  <li key={file.id} className={`file-item ${file.status}`}>
                    <span className="file-name">{file.name}</span>
                    <span className="file-status">
                      {file.status === 'processing' && 'Processing...'}
                      {file.status === 'success' && `Processed (${file.chunks || 0} chunks)`}
                      {file.status === 'error' && 'Error'}
                    </span>
                    <div className="file-details">
                      <span className="file-rag-type">
                        {file.ragType && `RAG: ${file.ragType}`}
                      </span>
                      <span className="file-chunking-strategy">
                        {file.chunkingStrategy && `Chunking: ${file.chunkingStrategy === 'basic' ? 'Basic' : 'Super'}`}
                      </span>
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          )}
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