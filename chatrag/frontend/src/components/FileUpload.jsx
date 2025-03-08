import React, { useState, useRef } from 'react';

const FileUpload = ({ userId, onFileUpload }) => {
  const [dragActive, setDragActive] = useState(false);
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef(null);

  // Handle file drop events
  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  // Handle actual file drop
  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFiles(e.dataTransfer.files);
    }
  };

  // Handle file input change
  const handleChange = (e) => {
    e.preventDefault();
    
    if (e.target.files && e.target.files[0]) {
      handleFiles(e.target.files);
    }
  };

  // Process selected files
  const handleFiles = async (files) => {
    setUploading(true);
    
    try {
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        
        // Create a file object to track status
        const fileObj = {
          id: `file-${Date.now()}-${i}`,
          name: file.name,
          size: file.size,
          type: file.type,
          status: 'processing',
        };
        
        // Notify parent component
        onFileUpload(fileObj);
        
        // Upload the file
        await uploadFile(file, fileObj.id);
      }
    } catch (error) {
      console.error('Error uploading files:', error);
    } finally {
      setUploading(false);
    }
  };

  // Upload a single file
  const uploadFile = async (file, fileId) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('user_id', userId);
    
    try {
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`Upload failed with status: ${response.status}`);
      }
      
      // Response will be handled by WebSocket notification
    } catch (error) {
      console.error('Upload error:', error);
      
      // Update file status to error
      onFileUpload({
        id: fileId,
        name: file.name,
        status: 'error',
      });
    }
  };

  // Trigger file input click
  const onButtonClick = () => {
    fileInputRef.current.click();
  };

  return (
    <div className="file-upload">
      <div 
        className={`dropzone ${dragActive ? 'active' : ''} ${uploading ? 'uploading' : ''}`}
        onDragEnter={handleDrag}
        onDragOver={handleDrag}
        onDragLeave={handleDrag}
        onDrop={handleDrop}
        onClick={onButtonClick}
      >
        <input
          ref={fileInputRef}
          type="file"
          id="file-upload"
          multiple
          onChange={handleChange}
          accept=".txt,.pdf,.docx,.doc,.csv,.xlsx,.xls"
          style={{ display: 'none' }}
        />
        
        {uploading ? (
          <div className="upload-status">
            <div className="spinner"></div>
            <p>Uploading...</p>
          </div>
        ) : (
          <div className="upload-prompt">
            <p>
              Drag & drop files or <span className="browse-link">browse</span>
            </p>
            <small>
              Supports: PDF, DOCX, TXT, CSV, XLSX
            </small>
          </div>
        )}
      </div>
    </div>
  );
};

export default FileUpload;