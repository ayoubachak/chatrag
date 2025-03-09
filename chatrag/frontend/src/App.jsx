// In App.jsx - Add a simple page switching mechanism
import React, { useState, useEffect } from 'react';
import ChatInterface from './components/ChatInterface';
import BenchmarkPage from './components/BenchmarkPage';
import ConnectionStatus from './components/ConnectionStatus';
import './styles/main.css';

const App = () => {
  const [isConnected, setIsConnected] = useState(false);
  const [userId, setUserId] = useState('');
  const [currentPage, setCurrentPage] = useState('chat');

  useEffect(() => {
    // Generate a random user ID on first load
    const newUserId = `user_${Math.random().toString(36).substring(2, 15)}`;
    setUserId(newUserId);
    
    // Check if URL has a hash for navigation
    const hash = window.location.hash.replace('#', '');
    if (hash === 'benchmark') {
      setCurrentPage('benchmark');
    }
  }, []);

  const navigateTo = (page) => {
    setCurrentPage(page);
    window.location.hash = page;
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>RAG Chat Application</h1>
        <nav className="app-nav">
          <button 
            onClick={() => navigateTo('chat')} 
            className={`nav-button ${currentPage === 'chat' ? 'active' : ''}`}
          >
            Chat
          </button>
          <button 
            onClick={() => navigateTo('benchmark')} 
            className={`nav-button ${currentPage === 'benchmark' ? 'active' : ''}`}
          >
            Benchmark
          </button>
        </nav>
        <ConnectionStatus isConnected={isConnected} />
      </header>
      
      <main className="app-main">
        {currentPage === 'chat' && userId && (
          <ChatInterface 
            userId={userId} 
            onConnectionChange={setIsConnected} 
          />
        )}
        {currentPage === 'benchmark' && (
          <BenchmarkPage />
        )}
      </main>
      
      <footer className="app-footer">
        <p>Built with FastAPI and React</p>
      </footer>
    </div>
  );
};

export default App;