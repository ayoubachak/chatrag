import React from 'react';

const RagSelector = ({ ragType, onChange }) => {
  const ragTypes = [
    { id: 'basic', name: 'Basic', description: 'Simple in-memory vector store' },
    { id: 'faiss', name: 'FAISS', description: 'Fast similarity search' },
    { id: 'chroma', name: 'ChromaDB', description: 'Persistent vector database' },
    { id: 'hybrid', name: 'Hybrid', description: 'FAISS + ChromaDB for optimal performance' }
  ];

  return (
    <div className="rag-selector">
      <select
        value={ragType}
        onChange={(e) => onChange(e.target.value)}
        className="rag-select"
      >
        {ragTypes.map((type) => (
          <option key={type.id} value={type.id}>
            {type.name}
          </option>
        ))}
      </select>
      <div className="rag-description">
        {ragTypes.find(type => type.id === ragType)?.description}
      </div>
    </div>
  );
};

export default RagSelector; 