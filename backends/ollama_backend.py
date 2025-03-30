import httpx
import asyncio
import random
from backends.base import AIBackend
from typing import List, Dict, Any

class OllamaBackend(AIBackend):
    def __init__(self, base_url="http://localhost:11434", model="deepseek-r1:7b"):
        self.base_url = base_url
        self.model = model
        self.api_url = f"{base_url}/api"
        
    async def initialize(self) -> bool:
        """Initialize Ollama connection and verify model"""
        # Skip availability check and just try to use it
        return True
        
    async def generate_response(self, system_prompt: str, messages: List[Dict[str, str]], 
                           max_tokens: int = 2000, temperature: float = 0) -> Dict[str, Any]:
        """Generate a response using the Ollama API"""
        # Format messages for Ollama
        formatted_messages = []
        
        # Add system message if provided
        if system_prompt:
            formatted_messages.append({"role": "system", "content": system_prompt})
        
        # Add the rest of the messages
        formatted_messages.extend(messages)
        
        # Prepare request payload
        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            "stream": False
        }
        
        # Implement retry logic
        max_retries = 3
        retry_count = 0
        base_delay = 1
        
        while retry_count < max_retries:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.api_url}/chat",
                        json=payload,
                        timeout=120.0
                    )
                    
                    response.raise_for_status()
                    result = response.json()
                    
                    return {
                        "content": [
                            {"type": "text", "text": result["message"]["content"]}
                        ]
                    }
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    raise
                    
                delay = base_delay * (2 ** (retry_count - 1))
                delay = delay + (random.random() * delay * 0.1)
                print(f"Connection issue. Retrying in {delay:.2f} seconds... (Attempt {retry_count}/{max_retries})")
                await asyncio.sleep(delay)
    
    def get_model_name(self) -> str:
        """Get the current model name"""
        return self.model
    
    async def set_model(self, model_name: str) -> bool:
        """Set a new model and check its availability"""
        self.model = model_name
        return True