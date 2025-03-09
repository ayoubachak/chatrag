import os
import time
import json
import logging
import random
from typing import List, Dict, Optional, Tuple
from pathlib import Path

logger = logging.getLogger("token_manager")

class TokenManager:
    """
    Manages API tokens with rotation and cooldown functionality.
    Automatically rotates tokens when they are exhausted or rate-limited.
    """
    
    def __init__(self, service_name: str, tokens_file: str = "tokens.json", cooldown_minutes: int = 60):
        """
        Initialize the token manager.
        
        Args:
            service_name: Name of the service (e.g., "huggingface", "openai")
            tokens_file: Path to the JSON file containing tokens
            cooldown_minutes: Cooldown period in minutes for exhausted tokens
        """
        self.service_name = service_name
        self.tokens_file = tokens_file
        self.cooldown_minutes = cooldown_minutes
        self.tokens = []  # List of (token, last_failure_time) tuples
        self.current_index = 0
        
        # Load tokens
        self._load_tokens()
        
    def _load_tokens(self):
        """Load tokens from the tokens file."""
        try:
            # Create tokens file if it doesn't exist
            tokens_path = Path(self.tokens_file)
            if not tokens_path.exists():
                # Create with empty structure
                with open(tokens_path, "w") as f:
                    json.dump({}, f, indent=2)
                logger.info(f"Created empty tokens file at {tokens_path}")
                
            # Load tokens
            with open(tokens_path, "r") as f:
                all_tokens = json.load(f)
                
            # Get tokens for this service
            service_tokens = all_tokens.get(self.service_name, [])
            
            # Convert to list of (token, last_failure_time) tuples
            self.tokens = [(token, 0) for token in service_tokens]
            
            if not self.tokens:
                # Try to get from environment variables as fallback
                env_token = os.environ.get(f"{self.service_name.upper()}_API_KEY")
                if env_token:
                    self.tokens = [(env_token, 0)]
                    logger.info(f"Loaded token from environment variable {self.service_name.upper()}_API_KEY")
                    
            logger.info(f"Loaded {len(self.tokens)} tokens for {self.service_name}")
            
            # Shuffle tokens to distribute load
            random.shuffle(self.tokens)
            
        except Exception as e:
            logger.error(f"Error loading tokens: {str(e)}")
            # Try to get from environment variables as fallback
            env_token = os.environ.get(f"{self.service_name.upper()}_API_KEY")
            if env_token:
                self.tokens = [(env_token, 0)]
                logger.info(f"Loaded token from environment variable {self.service_name.upper()}_API_KEY")
    
    def add_token(self, token: str):
        """
        Add a new token to the rotation.
        
        Args:
            token: The API token to add
        """
        # Check if token already exists
        if any(t[0] == token for t in self.tokens):
            return
            
        # Add token
        self.tokens.append((token, 0))
        logger.info(f"Added new token for {self.service_name}")
        
        # Save to file
        self._save_tokens()
    
    def _save_tokens(self):
        """Save tokens to the tokens file."""
        try:
            # Load existing tokens
            with open(self.tokens_file, "r") as f:
                all_tokens = json.load(f)
                
            # Update tokens for this service
            all_tokens[self.service_name] = [token for token, _ in self.tokens]
            
            # Save tokens
            with open(self.tokens_file, "w") as f:
                json.dump(all_tokens, f, indent=2)
                
            logger.info(f"Saved {len(self.tokens)} tokens for {self.service_name}")
            
        except Exception as e:
            logger.error(f"Error saving tokens: {str(e)}")
    
    def get_token(self) -> Optional[str]:
        """
        Get the next available token.
        
        Returns:
            An API token or None if no tokens are available
        """
        if not self.tokens:
            return None
            
        # Try to find a token that's not in cooldown
        now = time.time()
        cooldown_seconds = self.cooldown_minutes * 60
        
        # Start from current index and try all tokens
        for _ in range(len(self.tokens)):
            token, last_failure = self.tokens[self.current_index]
            
            # Check if token is in cooldown
            if now - last_failure < cooldown_seconds:
                # Token is in cooldown, try next one
                self.current_index = (self.current_index + 1) % len(self.tokens)
                continue
                
            # Token is available
            return token
            
        # All tokens are in cooldown
        logger.warning(f"All tokens for {self.service_name} are in cooldown")
        
        # Return the token with the oldest failure time
        self.tokens.sort(key=lambda x: x[1])
        return self.tokens[0][0]
    
    def report_success(self, token: str):
        """
        Report a successful API call with the token.
        
        Args:
            token: The token that was used successfully
        """
        # Nothing to do for now, but could be used for metrics
        pass
    
    def report_failure(self, token: str, is_rate_limit: bool = False):
        """
        Report a failed API call with the token.
        
        Args:
            token: The token that failed
            is_rate_limit: Whether the failure was due to rate limiting
        """
        # Find the token
        for i, (t, _) in enumerate(self.tokens):
            if t == token:
                # Update last failure time
                self.tokens[i] = (t, time.time())
                
                # Move to next token
                self.current_index = (i + 1) % len(self.tokens)
                
                logger.warning(f"Token for {self.service_name} marked as failed{' (rate limit)' if is_rate_limit else ''}")
                break
    
    def rotate_token(self):
        """
        Rotate to the next token regardless of status.
        """
        if not self.tokens:
            return
            
        # Move to next token
        self.current_index = (self.current_index + 1) % len(self.tokens)
        logger.info(f"Rotated to next token for {self.service_name}")
        
    @property
    def has_tokens(self) -> bool:
        """Check if there are any tokens available."""
        return len(self.tokens) > 0
        
    @property
    def token_count(self) -> int:
        """Get the number of tokens."""
        return len(self.tokens) 