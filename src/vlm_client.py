"""
Multi-Provider VLM Client for Grok, Qwen (vLLM), and Gemini APIs
"""

import os
import time
import requests
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv
from abc import ABC, abstractmethod

@dataclass
class VLMConfig:
    """Configuration for VLM API"""
    api_key: Optional[str] = None
    model: str = "grok-4-fast"
    api_base: str = "https://api.x.ai/v1"
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: int = 240
    max_retries: int = 1
    retry_delay: float = 1.0

class BaseVLMClient(ABC):
    """Base class for VLM clients"""
    
    def __init__(self, config: VLMConfig):
        self.config = config
        self.session = requests.Session()
        self._setup_session()
    
    @abstractmethod
    def _setup_session(self):
        """Setup session headers and auth"""
        pass
    
    @abstractmethod
    def _build_payload(self, prompt: str, system_prompt: Optional[str]) -> dict:
        """Build API request payload"""
        pass
    
    @abstractmethod
    def _extract_response(self, data: dict) -> str:
        """Extract text from API response"""
        pass
    
    def query(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Query the VLM API with a prompt
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt for context
            
        Returns:
            Response text from the model
        """
        payload = self._build_payload(prompt, system_prompt)
        
        for attempt in range(self.config.max_retries):
            try:
                response = self.session.post(
                    f'{self.config.api_base}/chat/completions',
                    json=payload,
                    timeout=self.config.timeout
                )
                
                response.raise_for_status()
                data = response.json()
                return self._extract_response(data)
                
            except requests.exceptions.Timeout:
                print(f"Timeout on attempt {attempt + 1}/{self.config.max_retries}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                    continue
                raise
            
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    print(f"Rate limited, waiting before retry {attempt + 1}/{self.config.max_retries}")
                    time.sleep(self.config.retry_delay * (attempt + 1) * 2)
                    continue
                elif e.response.status_code >= 500:
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


class GrokClient(BaseVLMClient):
    """Client for Grok API"""
    
    def _setup_session(self):
        self.session.headers.update({
            'Authorization': f'Bearer {self.config.api_key}',
            'Content-Type': 'application/json'
        })
    
    def _build_payload(self, prompt: str, system_prompt: Optional[str]) -> dict:
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': prompt})
        
        return {
            'model': self.config.model,
            'messages': messages,
            'max_tokens': self.config.max_tokens,
            'temperature': self.config.temperature
        }
    
    def _extract_response(self, data: dict) -> str:
        if 'choices' in data and len(data['choices']) > 0:
            return data['choices'][0]['message']['content']
        raise ValueError(f"Unexpected response format: {data}")


class QwenClient(BaseVLMClient):
    """Client for Qwen via vLLM (local)"""
    
    def _setup_session(self):
        self.session.headers.update({
            'Content-Type': 'application/json'
        })
    
    def _build_payload(self, prompt: str, system_prompt: Optional[str]) -> dict:
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': prompt})
        
        return {
            'model': self.config.model,
            'messages': messages,
            'max_tokens': self.config.max_tokens,
            'temperature': self.config.temperature
        }
    
    def _extract_response(self, data: dict) -> str:
        if 'choices' in data and len(data['choices']) > 0:
            return data['choices'][0]['message']['content']
        raise ValueError(f"Unexpected response format: {data}")


class GeminiClient(BaseVLMClient):
    """Client for Google Gemini API"""
    
    def _setup_session(self):
        self.session.headers.update({
            'Content-Type': 'application/json'
        })
    
    def _build_payload(self, prompt: str, system_prompt: Optional[str]) -> dict:
        # Gemini uses different format
        contents = []
        
        # Combine system prompt with user prompt for Gemini
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        contents.append({
            'parts': [{'text': full_prompt}]
        })
        
        return {
            'contents': contents,
            'generationConfig': {
                'temperature': self.config.temperature,
                'maxOutputTokens': self.config.max_tokens,
            }
        }
    
    def _extract_response(self, data: dict) -> str:
        if 'candidates' in data and len(data['candidates']) > 0:
            parts = data['candidates'][0]['content']['parts']
            return parts[0]['text']
        raise ValueError(f"Unexpected response format: {data}")
    
    def query(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Override to use Gemini's specific endpoint"""
        payload = self._build_payload(prompt, system_prompt)
        
        # Gemini uses different endpoint structure
        endpoint = f'{self.config.api_base}/models/{self.config.model}:generateContent?key={self.config.api_key}'
        
        for attempt in range(self.config.max_retries):
            try:
                response = self.session.post(
                    endpoint,
                    json=payload,
                    timeout=self.config.timeout
                )
                
                response.raise_for_status()
                data = response.json()
                return self._extract_response(data)
                
            except Exception as e:
                print(f"Error on attempt {attempt + 1}/{self.config.max_retries}: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                    continue
                raise
        
        raise Exception(f"Failed after {self.config.max_retries} retries")


def create_client(provider: str = "grok", config: Optional[VLMConfig] = None) -> BaseVLMClient:
    """
    Factory function to create appropriate VLM client
    
    Args:
        provider: One of "grok", "qwen", or "gemini"
        config: Optional custom config (if None, uses defaults from env)
        
    Returns:
        Configured VLM client
    """
    load_dotenv()
    provider = provider.lower()
    
    if provider == "grok":
        if config is None:
            api_key = os.getenv('GROK_API_KEY')
            if not api_key:
                raise ValueError("GROK_API_KEY not set")
            config = VLMConfig(
                api_key=api_key,
                model="grok-4-fast",
                api_base="https://api.x.ai/v1"
            )
        return GrokClient(config)
    
    elif provider == "qwen":
        if config is None:
            config = VLMConfig(
                api_key=None,  # No API key needed for local
                model="Qwen/Qwen2.5-7B-Instruct",
                api_base="http://localhost:8000/v1"
            )
        return QwenClient(config)
    
    elif provider == "gemini":
        if config is None:
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("GEMINI_API_KEY not set")
            config = VLMConfig(
                api_key=api_key,
                model="gemini-1.5-flash",
                api_base="https://generativelanguage.googleapis.com/v1beta"
            )
        return GeminiClient(config)
    
    else:
        raise ValueError(f"Unknown provider: {provider}. Choose 'grok', 'qwen', or 'gemini'")


# Example usage
if __name__ == "__main__":
    # Choose your provider here
    PROVIDER = "grok"  # Change to "qwen" or "gemini"
    
    try:
        client = create_client(PROVIDER)
        
        response = client.query(
            "What is 2+2?",
            system_prompt="You are a helpful assistant."
        )
        
        print(f"\n[{PROVIDER.upper()}] Response:", response)
        
    except ValueError as e:
        print(f"Error: {e}")
        print("\nSetup instructions:")
        print("- Grok: export GROK_API_KEY=your_key")
        print("- Qwen: Start vLLM server (see instructions below)")
        print("- Gemini: export GEMINI_API_KEY=your_key")
        
        print("\n--- vLLM Setup for Qwen ---")
        print("1. Install: pip install vllm")
        print("2. Start server: python -m vllm.entrypoints.openai.api_server \\")
        print("     --model Qwen/Qwen2.5-7B-Instruct \\")
        print("     --port 8000")
        print("3. Server runs at: http://localhost:8000")