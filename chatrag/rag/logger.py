import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Create logs directory if it doesn't exist
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Configure the root logger
def setup_logger(name, log_level=logging.INFO):
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Name of the logger
        log_level: Logging level
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Remove existing handlers if any
    if logger.handlers:
        logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # Create file handler
    today = datetime.now().strftime('%Y-%m-%d')
    log_file = logs_dir / f"{name}_{today}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Create loggers for different components
rag_logger = setup_logger("rag")
embedding_logger = setup_logger("embedding")
retriever_logger = setup_logger("retriever")
chat_logger = setup_logger("chat")
files_logger = setup_logger("files") 