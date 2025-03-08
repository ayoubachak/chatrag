import React from 'react';

const ModelSelector = ({ modelType, onChange }) => {
  const models = [
    {
      id: 'local',
      name: 'Local Model',
      description: 'Use a local model for generation',
      icon: '💻'
    },
    {
      id: 'huggingface',
      name: 'HuggingFace API',
      description: 'Use HuggingFace API for generation',
      icon: '🤗'
    },
    {
      id: 'lm_studio',
      name: 'LM Studio',
      description: 'Use LM Studio local API',
      icon: '🧠'
    }
  ];

  return (
    <div className="model-selector">
      {models.map((model) => (
        <button
          key={model.id}
          className={`model-button ${modelType === model.id ? 'active' : ''}`}
          onClick={() => onChange(model.id)}
        >
          <div className="model-icon">{model.icon}</div>
          <div className="model-info">
            <div className="model-name">{model.name}</div>
            <div className="model-description">{model.description}</div>
          </div>
        </button>
      ))}
    </div>
  );
};

export default ModelSelector;