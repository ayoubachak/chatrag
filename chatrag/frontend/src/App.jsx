import React, { useState, useEffect } from 'react';
import ChatInterface from './components/ChatInterface';
import ConnectionStatus from './components/ConnectionStatus';
import './styles/main.css';

const App = () => {
  const [isConnected, setIsConnected] = useState(false);
  const [userId, setUserId] = useState('');

  useEffect(() => {
    // Generate a random user ID on first load
    const newUserId = `user_${Math.random().toString(36).substring(2, 15)}`;
    setUserId(newUserId);
  }, []);

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>RAG Chat Application</h1>
        <ConnectionStatus isConnected={isConnected} />
      </header>
      
      <main className="app-main">
        {userId && (
          <ChatInterface 
            userId={userId} 
            onConnectionChange={setIsConnected} 
          />
        )}
      </main>
      
      <footer className="app-footer">
        <p>Built with FastAPI and React</p>
      </footer>
    </div>
  );
};

export default App;