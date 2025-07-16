import os
import logging
import requests
from typing import Dict, Any, List
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class ASI1Client:
    """Client for ASI1-mini API."""
    
    def __init__(self):
        self.api_key = os.getenv("ASI1_API_KEY")
        self.api_url = "https://api.asi1.ai/v1/chat/completions"
        
        if not self.api_key:
            logger.warning("ASI1 API key not found. Set ASI1_API_KEY environment variable.")
    
    def get_completion(self, 
                      messages: List[Dict[str, str]], 
                      temperature: float = 0.2, 
                      stream: bool = False) -> Dict[str, Any]:
        """Get completion from ASI1-mini model."""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": "asi1-mini",
                "messages": messages,
                "temperature": temperature,
                "stream": stream
            }
            
            response = requests.post(
                self.api_url, 
                headers=headers, 
                json=payload
            )
            
            response.raise_for_status()
            
            return response.json()
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling ASI1 API: {e}")
            return {"error": str(e)}