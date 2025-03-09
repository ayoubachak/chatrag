import React from 'react';

const ChunkingSelector = ({ chunkingStrategy, onChange }) => {
  const strategies = [
    { 
      id: 'basic', 
      name: 'Basic Chunking', 
      description: 'Simple chunking by paragraphs and pages' 
    },
    { 
      id: 'super', 
      name: 'Super Chunking', 
      description: 'Advanced semantic chunking with overlap' 
    }
  ];

  return (
    <div className="chunking-selector">
      <select
        value={chunkingStrategy}
        onChange={(e) => onChange(e.target.value)}
        className="chunking-select"
      >
        {strategies.map((strategy) => (
          <option key={strategy.id} value={strategy.id}>
            {strategy.name}
          </option>
        ))}
      </select>
      <div className="chunking-description">
        {strategies.find(strategy => strategy.id === chunkingStrategy)?.description}
      </div>
    </div>
  );
};

export default ChunkingSelector; 