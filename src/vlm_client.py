"""
Multi-Provider VLM Client for OpenAI-compatible APIs and Gemini

Supports:
- OpenAI-compatible providers (Grok, Qwen, OpenAI, Claude, etc.) via OpenAICompatibleClient
- Google Gemini via GeminiClient (different API structure)
"""

import os
import time
import requests
import threading
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv
from abc import ABC, abstractmethod

class RateLimiter:
    """Simple thread-safe rate limiter"""
    def __init__(self, max_rpm: int):
        self.interval = 60.0 / max_rpm
        self.last_call_time = 0
        self._lock = threading.Lock()

    def wait(self):
        with self._lock:
            now = time.time()
            elapsed = now - self.last_call_time
            wait_time = self.interval - elapsed
            if wait_time > 0:
                time.sleep(wait_time)
                self.last_call_time = time.time()
            else:
                self.last_call_time = now

@dataclass
class VLMConfig:
    """Configuration for VLM API"""
    api_key: Optional[str] = None
    model: str = "grok-4-fast"
    api_base: str = "https://api.x.ai/v1"
    max_tokens: int = 32096
    temperature: float = 0.7
    timeout: int = 480
    max_retries: int = 1
    retry_delay: float = 60.0
    save_prompts: bool = False
    prompt_log_dir: str = "prompts"
    suppress_errors: bool = False  # Return empty string on errors instead of raising
    max_rpm: int = 450

class BaseVLMClient(ABC):
    """Base class for VLM clients"""
    
    def __init__(self, config: VLMConfig):
        self.config = config
        self.session = requests.Session()
        self._setup_session()
        self.prompt_counter = 0
        self._lock = threading.Lock()
        self.rate_limiter = RateLimiter(config.max_rpm)
    
    @abstractmethod
    def _setup_session(self):
        """Setup session headers and auth"""
        pass
    
    @abstractmethod
    def _build_payload(self, prompt: Union[str, List[Dict[str, Any]]], system_prompt: Optional[str]) -> dict:#TODO: Change system prompt to union of str and content blocks
        """Build API request payload"""
        pass
    
    @abstractmethod
    def _extract_response(self, data: dict) -> str:
        """Extract text from API response"""
        pass
    
    def _save_prompt_html(self, payload: dict, prompt_type: str = "query"):
        """Save prompt as HTML file with images rendered inline"""
        if not self.config.save_prompts:
            return
        
        # Create log directory
        log_dir = Path(self.config.prompt_log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        with self._lock:
            self.prompt_counter += 1
            current_counter = self.prompt_counter
            
        filename = f"{prompt_type}_{current_counter:03d}.html"
        filepath = log_dir / filename
        
        # Build HTML
        html_parts = ['<!DOCTYPE html><html><head><meta charset="utf-8">']
        html_parts.append('<style>body{font-family:monospace;margin:20px;} ')
        html_parts.append('.message{border:1px solid #ccc;margin:10px 0;padding:10px;} ')
        html_parts.append('.role{font-weight:bold;color:#0066cc;} ')
        html_parts.append('img{max-width:600px;margin:10px 0;border:1px solid #eee;}</style></head><body>')
        html_parts.append(f'<h2>Prompt Log #{current_counter}</h2>')
        
        # Extract messages from payload
        messages = payload.get('messages', [])
        
        for msg in messages:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            
            html_parts.append(f'<div class="message"><div class="role">{role.upper()}:</div>')
            
            # Handle string content
            if isinstance(content, str):
                html_parts.append(f'<pre>{content}</pre>')
            
            # Handle content blocks (list)
            elif isinstance(content, list):
                for block in content:
                    if block.get('type') == 'text':
                        html_parts.append(f'<pre>{block["text"]}</pre>')
                    elif block.get('type') == 'image_url':
                        url = block['image_url']['url']
                        html_parts.append(f'<img src="{url}" />')
            
            html_parts.append('</div>')
        
        html_parts.append('</body></html>')
        
        # Write file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(html_parts))
        
        print(f"📝 Saved prompt to: {filepath}")
        
    def query(self, prompt: Union[str, List[Dict[str, Any]]], system_prompt: Optional[str] = None) -> str:
        """
        Query the VLM API with a prompt

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt for context

        Returns:
            Response text from the model (or empty string if suppress_errors=True and error occurs)
        """
        self.rate_limiter.wait()
        try:
            payload = self._build_payload(prompt, system_prompt)
            # self._save_prompt_html(payload, prompt_type=f"{self.config.model}")

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

        except Exception as e:
            if self.config.suppress_errors:
                return ""
            raise


class OpenAICompatibleClient(BaseVLMClient):
    """Client for OpenAI-compatible APIs (Grok, Qwen, Claude, GPT, etc.)"""

    def _setup_session(self):
        headers = {'Content-Type': 'application/json'}
        # Add API key if provided (not needed for local servers like Qwen)
        if self.config.api_key:
            headers['Authorization'] = f'Bearer {self.config.api_key}'
        self.session.headers.update(headers)

    def _build_payload(self, prompt: Union[str, List[Dict[str, Any]]], system_prompt: Optional[str]) -> dict:
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
    
    def _build_payload(self, prompt: Union[str, List[Dict[str, Any]]], system_prompt: Optional[str]) -> dict:
        contents = []
        
        # Combine system prompt with user prompt
        if isinstance(prompt, str):
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            contents.append({'parts': [{'text': full_prompt}]})
        else:  # List of content blocks
            parts = []
            if system_prompt:
                parts.append({'text': system_prompt})
            
            # Convert content blocks to Gemini format
            for block in prompt:
                if block.get('type') == 'text':
                    parts.append({'text': block['text']})
                elif block.get('type') == 'image_url':
                    # Extract base64 data
                    url = block['image_url']['url']
                    if url.startswith('data:image/png;base64,'):
                        base64_data = url.split(',', 1)[1]
                        parts.append({
                            'inline_data': {
                                'mime_type': 'image/png',
                                'data': base64_data
                            }
                        })
            contents.append({'parts': parts})
        
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
        try:
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

        except Exception as e:
            if self.config.suppress_errors:
                return ""
            raise


def create_client(provider: str = "grok", config: Optional[VLMConfig] = None) -> BaseVLMClient:
    """
    Factory function to create appropriate VLM client

    Args:
        provider: One of "grok", "qwen", "gemini", or any OpenAI-compatible provider
        config: Optional custom config (if None, uses defaults from env)

    Returns:
        Configured VLM client
    """
    load_dotenv()
    provider = provider.lower()

    # OpenAI-compatible providers (Grok, Qwen, etc.)
    if provider in ["grok", "qwen", "openai", "claude"]:
        if config is None:
            if provider == "grok":
                api_key = os.getenv('GROK_API_KEY')
                if not api_key:
                    raise ValueError("GROK_API_KEY not set")
                config = VLMConfig(
                    api_key=api_key,
                    model="grok-4-fast",
                    api_base="https://api.x.ai/v1"
                )
            elif provider == "qwen":
                config = VLMConfig(
                    api_key=None,  # No API key needed for local
                    model="Qwen/Qwen2.5-7B-Instruct",
                    api_base="http://localhost:8000/v1"
                )
        return OpenAICompatibleClient(config)

    # Gemini has different API structure
    elif provider == "gemini":
        if config is None:
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("GEMINI_API_KEY not set")
            config = VLMConfig(
                api_key=api_key,
                model="gemini-2.5-pro",
                api_base="https://generativelanguage.googleapis.com/v1beta"
            )
        return GeminiClient(config)

    else:
        raise ValueError(f"Unknown provider: {provider}. Choose 'grok', 'qwen', 'gemini', or other OpenAI-compatible provider")


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