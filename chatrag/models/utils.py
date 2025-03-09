from typing import List, Dict, Any, Optional

def format_chat_history(
    messages: List[Dict[str, Any]], 
    system_prompt: Optional[str] = None,
    current_prompt: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Format chat history into a properly structured sequence for LLMs.
    
    This ensures messages follow the pattern:
    1. One system message (if provided)
    2. Alternating user and assistant messages
    
    Args:
        messages: List of message objects with 'role' and 'content'
        system_prompt: Optional system prompt to use
        current_prompt: Current user prompt to append
        
    Returns:
        Properly formatted list of messages
    """
    # Initialize with system message if provided
    formatted_messages = []
    if system_prompt:
        formatted_messages.append({"role": "system", "content": system_prompt})
    
    # Filter out system messages and keep only user/assistant messages
    chat_messages = [msg for msg in messages if msg.get("role") in ["user", "assistant"]]
    
    # Ensure proper alternation (should start with user)
    filtered_messages = []
    expected_role = "user"
    
    for msg in chat_messages:
        role = msg.get("role")
        content = msg.get("content", "").strip()
        
        # Skip empty messages
        if not content:
            continue
            
        # If this message matches the expected role, add it
        if role == expected_role:
            filtered_messages.append({"role": role, "content": content})
            # Toggle expected role
            expected_role = "assistant" if expected_role == "user" else "user"
        # If we have consecutive messages of the same role, combine them
        elif filtered_messages and filtered_messages[-1]["role"] == role:
            filtered_messages[-1]["content"] += f"\n\n{content}"
        # If role doesn't match but we need to maintain sequence, skip
        # (This helps maintain the alternating pattern)
    
    # Add filtered messages to our result
    formatted_messages.extend(filtered_messages)
    
    # Add the current prompt if provided
    if current_prompt:
        # If the last message was from the user and we're adding another user message,
        # combine them to maintain alternation
        if formatted_messages and formatted_messages[-1]["role"] == "user":
            formatted_messages[-1]["content"] += f"\n\n{current_prompt}"
        else:
            formatted_messages.append({"role": "user", "content": current_prompt})
    
    return formatted_messages 