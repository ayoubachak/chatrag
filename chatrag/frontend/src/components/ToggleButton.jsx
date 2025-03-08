import React from 'react';

const ToggleButton = ({ label, active, onChange, icon }) => {
  return (
    <button
      className={`toggle-button ${active ? 'active' : ''}`}
      onClick={() => onChange(!active)}
      aria-pressed={active}
    >
      {icon && <span className="toggle-icon">{icon}</span>}
      <span className="toggle-label">{label}</span>
      <div className="toggle-switch">
        <div className="toggle-handle"></div>
      </div>
    </button>
  );
};

export default ToggleButton;