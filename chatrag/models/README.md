# LLM Models

This directory contains the implementation of various language models used in the RAG system.

## Model Types

- **LocalModel**: Uses local models loaded with Hugging Face's Transformers library
- **HuggingFaceModel**: Uses the OpenAI-compatible API provided by Hugging Face
- **LMStudioModel**: Uses the OpenAI-compatible API provided by LM Studio

## Token Manager

The `TokenManager` class provides token rotation and cooldown functionality for API-based models. This is particularly useful for the HuggingFace model, which may have rate limits or token exhaustion issues.

### Features

- **Token Rotation**: Automatically rotates to the next token when one is exhausted
- **Cooldown**: Puts exhausted tokens in cooldown for a specified period
- **Fallback**: Falls back to environment variables if no tokens are available
- **Persistence**: Saves tokens to a JSON file for reuse

### Usage

To add tokens to the token manager, use the `scripts/add_token.py` script:

```bash
python scripts/add_token.py --service huggingface --token YOUR_API_TOKEN
```

You can also add multiple tokens:

```bash
python scripts/add_token.py --service huggingface --token TOKEN1
python scripts/add_token.py --service huggingface --token TOKEN2
```

### Configuration

Tokens are stored in a JSON file (`tokens.json` by default) with the following structure:

```json
{
  "huggingface": [
    "token1",
    "token2"
  ],
  "openai": [
    "token3"
  ]
}
```

## Message Formatting

The `utils.py` file contains a utility function `format_chat_history` that ensures messages follow the proper sequence for LLMs:

1. One system message (if provided)
2. Alternating user and assistant messages

This ensures consistent behavior across different model implementations. 