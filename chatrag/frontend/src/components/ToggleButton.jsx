import React from 'react';

const ToggleButton = ({ label, active, onChange, icon, isActive, onClick, activeLabel, inactiveLabel }) => {
  // Support both old and new prop formats
  const isActiveState = isActive !== undefined ? isActive : active;
  const handleClick = onClick || (onChange ? () => onChange(!isActiveState) : () => {});
  
  return (
    <button
      className={`toggle-button ${isActiveState ? 'active' : ''}`}
      onClick={handleClick}
      aria-pressed={isActiveState}
    >
      {icon && <span className="toggle-icon">{icon}</span>}
      <span className="toggle-label">
        {label || (isActiveState ? activeLabel : inactiveLabel)}
      </span>
      <div className="toggle-switch">
        <div className="toggle-handle"></div>
      </div>
    </button>
  );
};

export default ToggleButton;