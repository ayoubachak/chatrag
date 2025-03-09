import React from 'react';
import ReactMarkdown from 'react-markdown';

const MessageList = ({ messages, isTyping }) => {
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

  return (
    <div className="message-list">
      {messages.length === 0 ? (
        <div className="welcome-message">
          <h2>Welcome to RAG Chat!</h2>
          <p>Upload files to use RAG or simply chat with the model.</p>
        </div>
      ) : (
        messages.map((message) => (
          <div
            key={message.id}
            className={`message ${message.role} ${message.isError ? 'error' : ''} ${message.isWarning ? 'warning' : ''} ${message.isSystemMessage ? 'system-message' : ''}`}
          >
            <div className="message-header">
              <span className="message-role">
                {message.role === 'user' ? '👤 You' : 
                 message.role === 'assistant' ? '🤖 Assistant' : '⚙️ System'}
              </span>
              {message.timestamp && (
                <span className="message-time">
                  {formatTimestamp(message.timestamp)}
                </span>
              )}
            </div>
            <div className="message-content">
              <ReactMarkdown>{message.content}</ReactMarkdown>
            </div>
            
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
        ))
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