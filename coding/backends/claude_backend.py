from anthropic import Anthropic
from backends.base import AIBackend
from typing import List, Dict, Any
import asyncio


class ClaudeBackend(AIBackend):
    def __init__(self, api_key=None, model="claude-3-5-sonnet-20241022"):
        self.api_key = api_key
        self.model = model
        self.client = None
        
    async def initialize(self) -> bool:
        """Initialize Claude client"""
        if not self.api_key:
            print("Error: ANTHROPIC_API_KEY not provided.")
            return False
            
        try:
            self.client = Anthropic(api_key=self.api_key)
            return True
        except Exception as e:
            print(f"Error initializing Claude: {str(e)}")
            return False
        
    async def generate_response(self, system_prompt: str, messages: List[Dict[str, str]], 
                           max_tokens: int = 2000, temperature: float = 0) -> Dict[str, Any]:
        """Generate a response using the Claude API"""
        if not self.client:
            raise Exception("Claude client not initialized. Call initialize() first.")
            
        max_retries = 3
        retry_count = 0
        base_delay = 2
        
        while retry_count < max_retries:
            try:
                response = self.client.messages.create(
                    system=system_prompt,
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=messages
                )
                
                # Return the response directly, no need for dict conversion
                return response
                    
            except Exception as e:
                error_str = str(e)
                
                # Handle retryable errors
                if "529" in error_str or "429" in error_str:
                    retry_count += 1
                    if retry_count >= max_retries:
                        raise
                        
                    import random
                    delay = base_delay * (2 ** (retry_count - 1))
                    delay = delay + (random.random() * delay * 0.1)
                    print(f"API issue. Retrying in {delay:.2f} seconds... (Attempt {retry_count}/{max_retries})")
                    await asyncio.sleep(delay)
                else:
                    # Non-retryable error
                    raise
    
    def get_model_name(self) -> str:
        """Get the current model name"""
        return self.model
    
    async def set_model(self, model_name: str) -> bool:
        """Set a new model"""
        self.model = model_name
        return True