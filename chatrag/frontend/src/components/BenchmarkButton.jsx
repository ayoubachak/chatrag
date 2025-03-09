import React, { useState } from 'react';

const BenchmarkButton = ({ userId, onBenchmarkResult }) => {
  const [running, setRunning] = useState(false);
  const [showOptions, setShowOptions] = useState(false);
  const [options, setOptions] = useState({
    documents: 100,
    queries: 20,
    implementations: ['basic', 'faiss', 'chroma', 'hybrid']
  });

  const handleOptionChange = (e) => {
    const { name, value, type, checked } = e.target;
    
    if (type === 'checkbox') {
      // Handle implementation checkboxes
      if (name.startsWith('impl-')) {
        const impl = name.replace('impl-', '');
        setOptions(prev => {
          const newImpls = checked 
            ? [...prev.implementations, impl]
            : prev.implementations.filter(i => i !== impl);
          return { ...prev, implementations: newImpls };
        });
      }
    } else {
      // Handle numeric inputs
      setOptions(prev => ({
        ...prev,
        [name]: parseInt(value, 10)
      }));
    }
  };

  const runBenchmark = async () => {
    setRunning(true);
    
    try {
      // Prepare query parameters
      const params = new URLSearchParams({
        user_id: userId,
        documents: options.documents,
        queries: options.queries,
        implementations: options.implementations.join(',')
      });
      
      // Call the benchmark API
      const response = await fetch(`/api/benchmark?${params.toString()}`);
      
      if (!response.ok) {
        throw new Error(`Benchmark failed with status: ${response.status}`);
      }
      
      const result = await response.json();
      
      // Notify parent component
      onBenchmarkResult(result);
      
    } catch (error) {
      console.error('Benchmark error:', error);
      onBenchmarkResult({ error: error.message });
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="benchmark-container">
      <button 
        className="benchmark-button"
        onClick={() => setShowOptions(!showOptions)}
        disabled={running}
      >
        {running ? 'Running Benchmark...' : '🔍 Benchmark RAG Implementations'}
      </button>
      
      {showOptions && (
        <div className="benchmark-options">
          <h4>Benchmark Options</h4>
          
          <div className="option-group">
            <label>
              Documents:
              <input 
                type="number" 
                name="documents"
                min="10"
                max="1000"
                value={options.documents}
                onChange={handleOptionChange}
                disabled={running}
              />
            </label>
          </div>
          
          <div className="option-group">
            <label>
              Queries:
              <input 
                type="number" 
                name="queries"
                min="5"
                max="100"
                value={options.queries}
                onChange={handleOptionChange}
                disabled={running}
              />
            </label>
          </div>
          
          <div className="option-group">
            <label>Implementations:</label>
            <div className="checkbox-group">
              {['basic', 'faiss', 'chroma', 'hybrid'].map(impl => (
                <label key={impl} className="checkbox-label">
                  <input 
                    type="checkbox"
                    name={`impl-${impl}`}
                    checked={options.implementations.includes(impl)}
                    onChange={handleOptionChange}
                    disabled={running}
                  />
                  {impl.charAt(0).toUpperCase() + impl.slice(1)}
                </label>
              ))}
            </div>
          </div>
          
          <button 
            className="run-benchmark-button"
            onClick={runBenchmark}
            disabled={running || options.implementations.length === 0}
          >
            {running ? 'Running...' : 'Run Benchmark'}
          </button>
          
          <p className="benchmark-note">
            Note: Benchmarking may take a few minutes to complete.
          </p>
        </div>
      )}
    </div>
  );
};

export default BenchmarkButton; 