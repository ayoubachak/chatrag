#!/usr/bin/env python3
"""
Script to add API tokens to the token manager.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add parent directory to path to import from models
sys.path.append(str(Path(__file__).parent.parent))
from models.token_manager import TokenManager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("add_token")

def main():
    parser = argparse.ArgumentParser(description="Add API tokens to the token manager")
    parser.add_argument("--service", "-s", required=True, help="Service name (e.g., huggingface, openai)")
    parser.add_argument("--token", "-t", required=True, help="API token to add")
    parser.add_argument("--file", "-f", default="tokens.json", help="Path to tokens file (default: tokens.json)")
    
    args = parser.parse_args()
    
    # Initialize token manager
    token_manager = TokenManager(args.service, args.file)
    
    # Add token
    token_manager.add_token(args.token)
    
    logger.info(f"Added token for {args.service}. Total tokens: {token_manager.token_count}")
    
if __name__ == "__main__":
    main() 