import React from 'react';

const BenchmarkResults = ({ results }) => {
  if (!results || Object.keys(results).length === 0) {
    return null;
  }

  // Handle error case
  if (results.error) {
    return (
      <div className="benchmark-results error">
        <h4>Benchmark Error</h4>
        <p>{results.error}</p>
      </div>
    );
  }

  // Extract data from results
  const { parameters, results: benchmarkResults } = results;
  
  // Find the fastest implementation for each metric
  const fastestAdd = Object.entries(benchmarkResults).reduce(
    (fastest, [impl, data]) => 
      data.add_time.total < fastest.time ? { impl, time: data.add_time.total } : fastest,
    { impl: '', time: Infinity }
  );
  
  const fastestQuery = Object.entries(benchmarkResults).reduce(
    (fastest, [impl, data]) => 
      data.query_time.average < fastest.time ? { impl, time: data.query_time.average } : fastest,
    { impl: '', time: Infinity }
  );
  
  const fastestSave = Object.entries(benchmarkResults).reduce(
    (fastest, [impl, data]) => 
      data.save_time < fastest.time ? { impl, time: data.save_time } : fastest,
    { impl: '', time: Infinity }
  );
  
  const fastestLoad = Object.entries(benchmarkResults).reduce(
    (fastest, [impl, data]) => 
      data.load_time < fastest.time ? { impl, time: data.load_time } : fastest,
    { impl: '', time: Infinity }
  );

  return (
    <div className="benchmark-results">
      <h4>Benchmark Results</h4>
      
      <div className="benchmark-parameters">
        <p>
          <strong>Documents:</strong> {parameters.num_documents} | 
          <strong> Queries:</strong> {parameters.num_queries} | 
          <strong> Batch Size:</strong> {parameters.batch_size} | 
          <strong> Top-K:</strong> {parameters.top_k}
        </p>
      </div>
      
      <table className="benchmark-table">
        <thead>
          <tr>
            <th>Implementation</th>
            <th>Add Time (s)</th>
            <th>Query Time (ms)</th>
            <th>Save Time (s)</th>
            <th>Load Time (s)</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(benchmarkResults).map(([impl, data]) => (
            <tr key={impl}>
              <td>{impl.charAt(0).toUpperCase() + impl.slice(1)}</td>
              <td className={fastestAdd.impl === impl ? 'benchmark-winner' : ''}>
                {data.add_time.total.toFixed(4)}
              </td>
              <td className={fastestQuery.impl === impl ? 'benchmark-winner' : ''}>
                {(data.query_time.average * 1000).toFixed(4)}
              </td>
              <td className={fastestSave.impl === impl ? 'benchmark-winner' : ''}>
                {data.save_time.toFixed(4)}
              </td>
              <td className={fastestLoad.impl === impl ? 'benchmark-winner' : ''}>
                {data.load_time.toFixed(4)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      
      <div className="benchmark-summary">
        <h4>Summary</h4>
        <p>
          <strong>Fastest document addition:</strong>{' '}
          <span className="benchmark-winner">{fastestAdd.impl}</span>
        </p>
        <p>
          <strong>Fastest query:</strong>{' '}
          <span className="benchmark-winner">{fastestQuery.impl}</span>
        </p>
        <p>
          <strong>Fastest save:</strong>{' '}
          <span className="benchmark-winner">{fastestSave.impl}</span>
        </p>
        <p>
          <strong>Fastest load:</strong>{' '}
          <span className="benchmark-winner">{fastestLoad.impl}</span>
        </p>
        
        <h4>Recommendation</h4>
        {fastestQuery.impl === 'faiss' && (
          <p>For speed-critical applications with many queries: Use <strong>FAISS</strong></p>
        )}
        {fastestQuery.impl === 'hybrid' && (
          <p>For balanced performance and persistence: Use <strong>Hybrid</strong></p>
        )}
        {fastestQuery.impl === 'chroma' && (
          <p>For persistence and advanced filtering: Use <strong>ChromaDB</strong></p>
        )}
        {fastestQuery.impl === 'basic' && (
          <p>For simple use cases: Use <strong>Basic</strong></p>
        )}
      </div>
    </div>
  );
};

export default BenchmarkResults; 