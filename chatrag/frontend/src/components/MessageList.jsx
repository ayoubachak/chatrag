import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';

const MessageList = ({ messages, isTyping }) => {
  // State to track expanded thinking sections
  const [expandedThinking, setExpandedThinking] = useState({});

  // Function to format timestamps
  const formatTimestamp = (timestamp) => {
    try {
      const date = new Date(timestamp);
      return date.toLocaleTimeString(undefined, {
        hour: '2-digit',
        minute: '2-digit'
      });
    } catch (e) {
      return '';
    }
  };

  // Function to extract thinking content and regular content
  const processThinkingContent = (content) => {
    if (!content || typeof content !== 'string') return { regular: content, thinking: null };

    const thinkRegex = /<think>([\s\S]*?)<\/think>/g;
    const matches = [...content.matchAll(thinkRegex)];
    
    if (matches.length === 0) return { regular: content, thinking: null };
    
    let regular = content;
    const thinking = matches.map(match => match[1].trim());
    
    // Remove thinking tags from regular content
    matches.forEach(match => {
      regular = regular.replace(match[0], '');
    });
    
    return { regular: regular.trim(), thinking };
  };

  // Toggle thinking expansion
  const toggleThinking = (messageId, thinkingIndex) => {
    setExpandedThinking(prev => {
      const key = `${messageId}-${thinkingIndex}`;
      return {
        ...prev,
        [key]: !prev[key]
      };
    });
  };

  return (
    <div className="message-list">
      {messages.length === 0 ? (
        <div className="welcome-message">
          <h2>Welcome to RAG Chat!</h2>
          <p>Upload files to use RAG or simply chat with the model.</p>
        </div>
      ) : (
        messages.map((message) => {
          const { regular, thinking } = processThinkingContent(message.content);
          const hasThinking = thinking && thinking.length > 0;

          return (
            <div
              key={message.id}
              className={`message ${message.role} ${message.isError ? 'error' : ''} ${message.isWarning ? 'warning' : ''} ${message.isSystemMessage ? 'system-message' : ''} ${message.isStreaming ? 'streaming' : ''}`}
            >
              <div className="message-header">
                <span className="message-role">
                  {message.role === 'user' ? '👤 You' : 
                   message.role === 'assistant' ? '🤖 Assistant' : '⚙️ System'}
                </span>
                {message.timestamp && (
                  <span className="message-time">
                    {formatTimestamp(message.timestamp)}
                    {message.isStreaming && <span className="streaming-indicator">streaming...</span>}
                  </span>
                )}
              </div>
              
              <div className="message-content">
                <ReactMarkdown>{regular}</ReactMarkdown>
                {message.isStreaming && (
                  <span className="cursor-blink">▌</span>
                )}
              </div>
              
              {/* Thinking process section */}
              {hasThinking && message.role === 'assistant' && (
                <div className="thinking-container">
                  {thinking.map((thought, index) => {
                    const thinkingKey = `${message.id}-${index}`;
                    const isExpanded = expandedThinking[thinkingKey];
                    
                    return (
                      <div key={thinkingKey} className="thinking-section">
                        <div 
                          className="thinking-header"
                          onClick={() => toggleThinking(message.id, index)}
                        >
                          <span className="thinking-icon">
                            {isExpanded ? '🧠 Hide thinking process' : '🧠 Show thinking process'}
                          </span>
                          <span className="thinking-toggle">
                            {isExpanded ? '▼' : '►'}
                          </span>
                        </div>
                        
                        {isExpanded && (
                          <div className="thinking-content">
                            <ReactMarkdown>{thought}</ReactMarkdown>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
              
              {/* Show sources if available from RAG */}
              {message.metadata?.sources?.length > 0 && (
                <div className="message-sources">
                  <div className="sources-header" onClick={() => {
                    const sourcesContent = document.getElementById(`sources-${message.id}`);
                    if (sourcesContent) {
                      sourcesContent.classList.toggle('expanded');
                    }
                  }}>
                    📚 Sources ({message.metadata.sources.length})
                  </div>
                  <div id={`sources-${message.id}`} className="sources-content">
                    {message.metadata.sources.map((source, index) => (
                      <div key={index} className="source-item">
                        <div className="source-metadata">
                          <span>Source {index + 1}</span>
                          <span>From: {source.metadata.source}</span>
                          {source.metadata.page && (
                            <span>Page: {source.metadata.page}</span>
                          )}
                          <span>Relevance: {Math.round(source.score * 100)}%</span>
                        </div>
                        <div className="source-content">
                          {source.content}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          );
        })
      )}
      
      {isTyping && (
        <div className="message assistant typing">
          <div className="typing-indicator">
            <span></span>
            <span></span>
            <span></span>
          </div>
        </div>
      )}
    </div>
  );
};

export default MessageList;