// BenchmarkPage.jsx - With custom CSS instead of Tailwind
import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import '../styles/BenchmarkPage.css'; // Import the custom CSS

const BenchmarkPage = () => {
  const [running, setRunning] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [options, setOptions] = useState({
    documents: 100,
    queries: 20,
    batchSize: 50,
    topK: 5,
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
    setError(null);
    
    try {
      // Prepare query parameters
      const params = new URLSearchParams({
        documents: options.documents,
        queries: options.queries,
        batch_size: options.batchSize,
        top_k: options.topK,
        implementations: options.implementations.join(',')
      });
      
      // Call the benchmark API (synchronous version)
      const response = await fetch(`/api/benchmark-sync?${params.toString()}`);
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Benchmark failed with status: ${response.status}`);
      }
      
      const resultData = await response.json();
      setResults(resultData);
      
    } catch (error) {
      console.error('Benchmark error:', error);
      setError(error.message);
    } finally {
      setRunning(false);
    }
  };

  // Prepare chart data if results are available
  const prepareChartData = () => {
    if (!results) return { addTimeData: [], queryTimeData: [], relevanceData: [] };
    
    const summary = results.summary || {};
    const implementations = Object.keys(results).filter(key => key !== 'summary');

    const addTimeData = implementations.map(impl => ({
      name: impl,
      time: results[impl].add_time?.total || 0,
      winner: impl === summary.fastest_add ? '⭐' : ''
    }));

    const queryTimeData = implementations.map(impl => ({
      name: impl,
      time: results[impl].query_time?.avg * 1000 || 0, // Convert to ms
      winner: impl === summary.fastest_query ? '⭐' : ''
    }));

    const relevanceData = implementations.map(impl => ({
      name: impl,
      score: results[impl].relevance?.avg * 100 || 0, // Convert to percentage
      winner: impl === summary.highest_relevance ? '⭐' : ''
    }));

    return { addTimeData, queryTimeData, relevanceData };
  };

  const { addTimeData, queryTimeData, relevanceData } = prepareChartData();
  const summary = results?.summary || {};
  const implementations = results ? Object.keys(results).filter(key => key !== 'summary') : [];

  // Format helpers
  const formatTime = (seconds) => {
    if (seconds === undefined || seconds === null) return 'N/A';
    const ms = Math.round(seconds * 1000 * 100) / 100;
    return `${ms.toFixed(2)} ms`;
  };

  const formatLargeTime = (seconds) => {
    if (seconds === undefined || seconds === null) return 'N/A';
    return `${seconds.toFixed(3)} s`;
  };

  const formatPercentage = (value) => {
    if (value === undefined || value === null) return 'N/A';
    return `${value.toFixed(2)}%`;
  };

  return (
    <div className="benchmark-container">
      <h1 className="page-title">RAG Implementation Benchmark</h1>
      
      {/* Benchmark Configuration */}
      <div className="config-section">
        <h2 className="section-title">Benchmark Configuration</h2>
        
        <div className="form-grid">
          <div>
            <div className="input-group">
              <label className="input-label">
                Documents:
                <input 
                  type="number" 
                  name="documents"
                  min="10"
                  max="1000"
                  value={options.documents}
                  onChange={handleOptionChange}
                  disabled={running}
                  className="input-field"
                />
              </label>
              <p className="input-hint">Number of test documents (10-1000)</p>
            </div>
            
            <div className="input-group">
              <label className="input-label">
                Queries:
                <input 
                  type="number" 
                  name="queries"
                  min="5"
                  max="100"
                  value={options.queries}
                  onChange={handleOptionChange}
                  disabled={running}
                  className="input-field"
                />
              </label>
              <p className="input-hint">Number of test queries (5-100)</p>
            </div>
          </div>
          
          <div>
            <div className="input-group">
              <label className="input-label">
                Batch Size:
                <input 
                  type="number" 
                  name="batchSize"
                  min="10"
                  max="200"
                  value={options.batchSize}
                  onChange={handleOptionChange}
                  disabled={running}
                  className="input-field"
                />
              </label>
              <p className="input-hint">Documents per batch (10-200)</p>
            </div>
            
            <div className="input-group">
              <label className="input-label">
                Top K:
                <input 
                  type="number" 
                  name="topK"
                  min="1"
                  max="20"
                  value={options.topK}
                  onChange={handleOptionChange}
                  disabled={running}
                  className="input-field"
                />
              </label>
              <p className="input-hint">Number of results per query (1-20)</p>
            </div>
          </div>
        </div>
        
        <div className="checkbox-group">
          <label className="input-label">Implementations:</label>
          <div className="checkbox-container">
            {['basic', 'faiss', 'chroma', 'hybrid'].map(impl => (
              <label key={impl} className="checkbox-label">
                <input 
                  type="checkbox"
                  name={`impl-${impl}`}
                  checked={options.implementations.includes(impl)}
                  onChange={handleOptionChange}
                  disabled={running}
                  className="checkbox-input"
                />
                <span className="capitalize">{impl}</span>
              </label>
            ))}
          </div>
        </div>
        
        <button 
          onClick={runBenchmark}
          disabled={running || options.implementations.length === 0}
          className={`btn ${
            running || options.implementations.length === 0 
              ? 'btn-disabled' 
              : 'btn-primary'
          }`}
        >
          {running ? 'Running Benchmark...' : 'Run Benchmark'}
        </button>
        
        {error && (
          <div className="error-message">
            <strong>Error:</strong> {error}
          </div>
        )}
      </div>
      
      {/* Results Section */}
      {results && (
        <div className="results-section">
          <h2 className="section-title">Benchmark Results</h2>
          
          {/* Summary Section */}
          <div className="summary-box">
            <h3 className="summary-title">Summary</h3>
            <div className="summary-grid">
              <div className="summary-card">
                <p>
                  <span className="summary-label">Fastest Add:</span>{' '}
                  <span className="capitalize">{summary.fastest_add || 'N/A'}</span>
                </p>
                <p>
                  <span className="summary-label">Fastest Query:</span>{' '}
                  <span className="capitalize">{summary.fastest_query || 'N/A'}</span>
                </p>
              </div>
              <div className="summary-card">
                <p>
                  <span className="summary-label">Highest Relevance:</span>{' '}
                  <span className="capitalize">{summary.highest_relevance || 'N/A'}</span>
                </p>
                <p>
                  <span className="summary-label">Overall Winner:</span>{' '}
                  <span className="text-green font-bold capitalize">{summary.overall_winner || 'N/A'}</span>
                </p>
              </div>
            </div>
          </div>
          
          {/* Charts Section */}
          <div className="charts-container">
            {/* Document Add Time Chart */}
            <div className="chart-box">
              <h3 className="chart-title">Document Add Time (seconds)</h3>
              <div className="chart-container">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={addTimeData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip formatter={(value) => value.toFixed(4) + ' s'} />
                    <Legend />
                    <Bar dataKey="time" name="Add Time (s)" fill="#4a6cf7" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
            
            {/* Query Time Chart */}
            <div className="chart-box">
              <h3 className="chart-title">Query Time (milliseconds)</h3>
              <div className="chart-container">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={queryTimeData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip formatter={(value) => value.toFixed(2) + ' ms'} />
                    <Legend />
                    <Bar dataKey="time" name="Query Time (ms)" fill="#4ade80" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
            
            {/* Relevance Score Chart */}
            <div className="chart-box">
              <h3 className="chart-title">Relevance Score (%)</h3>
              <div className="chart-container">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={relevanceData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis domain={[0, 100]} />
                    <Tooltip formatter={(value) => value.toFixed(2) + '%'} />
                    <Legend />
                    <Bar dataKey="score" name="Relevance Score (%)" fill="#f97316" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
          
          {/* Detailed Results Table */}
          <div className="table-container">
            <h3 className="table-title">Detailed Results</h3>
            <div className="table-wrapper">
              <table className="results-table">
                <thead className="table-header">
                  <tr>
                    <th>Implementation</th>
                    <th className="text-right">Add Time (s)</th>
                    <th className="text-right">Query Time (ms)</th>
                    <th className="text-right">Relevance Score (%)</th>
                    <th className="text-right">Documents</th>
                    <th className="text-right">Queries</th>
                  </tr>
                </thead>
                <tbody className="table-body">
                  {implementations.map(impl => (
                    <tr key={impl} className={impl === summary.overall_winner ? 'winner-row' : ''}>
                      <td className="table-cell">
                        {impl === summary.overall_winner && '⭐ '}
                        <span className="capitalize">{impl}</span>
                      </td>
                      <td className={`table-cell text-right ${impl === summary.fastest_add ? 'text-blue font-bold' : 'text-gray'}`}>
                        {formatLargeTime(results[impl].add_time?.total || 0)}
                      </td>
                      <td className={`table-cell text-right ${impl === summary.fastest_query ? 'text-green font-bold' : 'text-gray'}`}>
                        {formatTime(results[impl].query_time?.avg || 0)}
                      </td>
                      <td className={`table-cell text-right ${impl === summary.highest_relevance ? 'text-orange font-bold' : 'text-gray'}`}>
                        {formatPercentage((results[impl].relevance?.avg || 0) * 100)}
                      </td>
                      <td className="table-cell text-right text-gray">
                        {results[impl].total_docs || 'N/A'}
                      </td>
                      <td className="table-cell text-right text-gray">
                        {results[impl].total_queries || 'N/A'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default BenchmarkPage;