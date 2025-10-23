"""
VLM Client for Grok API
"""

import os
import time
import requests
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

@dataclass
class VLMConfig:
    """Configuration for VLM API"""
    api_key: str
    model: str = "grok-4-fast"  # Grok model name
    api_base: str = "https://api.x.ai/v1"
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: int = 60
    max_retries: int = 3
    retry_delay: int = 2

class VLMClient:
    """Client for Grok API interactions"""
    
    def __init__(self, config: VLMConfig = None):
        """
        Initialize VLM client
        
        Args:
            config: Optional VLMConfig, defaults to environment variables
        """
        if config is None:
            load_dotenv()
            api_key = os.getenv('GROK_API_KEY')
            if not api_key:
                raise ValueError(
                    "GROK_API_KEY environment variable not set. "
                    "Set it with: export GROK_API_KEY=your_key"
                )
            config = VLMConfig(api_key=api_key)
        
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.config.api_key}',
            'Content-Type': 'application/json'
        })
    
    def query(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Query Grok API with a prompt
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt for context
            
        Returns:
            Response text from Grok
            
        Raises:
            Exception: If API call fails after retries
        """
        messages = []
        
        if system_prompt:
            messages.append({
                'role': 'system',
                'content': system_prompt
            })
        
        messages.append({
            'role': 'user',
            'content': prompt
        })
        
        payload = {
            'model': self.config.model,
            'messages': messages,
            'max_tokens': self.config.max_tokens,
            'temperature': self.config.temperature
        }
        
        # Retry logic
        for attempt in range(self.config.max_retries):
            try:
                response = self.session.post(
                    f'{self.config.api_base}/chat/completions',
                    json=payload,
                    timeout=self.config.timeout
                )
                
                response.raise_for_status()
                
                data = response.json()
                
                # Extract response text
                if 'choices' in data and len(data['choices']) > 0:
                    content = data['choices'][0]['message']['content']
                    return content
                else:
                    raise ValueError(f"Unexpected response format: {data}")
                
            except requests.exceptions.Timeout:
                print(f"Timeout on attempt {attempt + 1}/{self.config.max_retries}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                    continue
                raise
            
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:  # Rate limit
                    print(f"Rate limited, waiting before retry {attempt + 1}/{self.config.max_retries}")
                    time.sleep(self.config.retry_delay * (attempt + 1) * 2)
                    continue
                elif e.response.status_code >= 500:  # Server error
                    print(f"Server error on attempt {attempt + 1}/{self.config.max_retries}")
                    if attempt < self.config.max_retries - 1:
                        time.sleep(self.config.retry_delay * (attempt + 1))
                        continue
                raise
            
            except Exception as e:
                print(f"Error on attempt {attempt + 1}/{self.config.max_retries}: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                    continue
                raise
        
        raise Exception(f"Failed after {self.config.max_retries} retries")


# Example usage
if __name__ == "__main__":
    # Test the client
    try:
        client = VLMClient()
        
        # Simple test query
        response = client.query(
            "What is 2+2?",
            system_prompt="You are a helpful assistant."
        )
        
        print("Response:", response)
        
    except ValueError as e:
        print(f"Error: {e}")
        print("\nTo use this client, set your Grok API key:")
        print("export GROK_API_KEY=your_key_here")