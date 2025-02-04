import asyncio
import logging
import os
import subprocess
import time
from typing import Any, Dict, Optional
from pydantic import BaseModel

import ollama
from tenacity import retry, stop_after_attempt, wait_exponential


class LLMResult(BaseModel):
    generated_text: Any
    input_language: Any
    model_name: str
    status: str
    

class LLMInferenceEngine:
    """
    A language model provider-agnostic inference engine that processes text input
    and generates output in the same language using Ollama.
    """
    
    def __init__(self, model_name: str = "llama3.2:1b", temperature: float = 0.7):
        """
        Initialize the LLM Inference Engine.
        
        Args:
            model_name (str): Name of the Ollama model to use
            temperature (float): Sampling temperature for text generation (0.0 to 1.0)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.logger = logging.getLogger(__name__)
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        
        # Ensure Ollama server is running and model is available
        self.start_ollama_server()
        self.check_and_pull_model()

    def start_ollama_server(self):
        """Start the Ollama server if it is not already running."""
        try:
            # Check if the server is already running
            response = subprocess.run(
                ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "http://127.0.0.1:11434"],
                capture_output=True,
                text=True
            )
            if response.returncode != 0 or response.stdout.strip() != "200":
                self.logger.info("Starting Ollama server...")
                subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self.logger.info("Ollama server started.")
            else:
                self.logger.info("Ollama server is already running.")
        except Exception as e:
            self.logger.error(f"Failed to start Ollama server: {str(e)}")

    def check_and_pull_model(self):
        """Check if the model is available and pull it if not."""
        try:
            # Use subprocess to list models since the Python client doesn't have list_models()
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Check if model exists in the output
            if self.model_name not in result.stdout:
                self.logger.info(f"Model {self.model_name} not found. Pulling the model...")
                # Pull the model and wait for it to complete
                pull_process = subprocess.run(
                    ["ollama", "pull", self.model_name],
                    capture_output=True,
                    text=True,
                    check=True
                )
                self.logger.info(f"Model {self.model_name} pulled successfully.")
            else:
                self.logger.info(f"Model {self.model_name} is already available.")
                
            # Wait a moment for the model to be fully loaded
            time.sleep(2)
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error executing ollama command: {str(e)}")
            raise RuntimeError(f"Failed to pull model: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error checking or pulling model: {str(e)}")
            raise RuntimeError(f"Failed to check/pull model: {str(e)}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_response(self, input_text: str, language: str, **kwargs) -> LLMResult:
        """
        Generate a response for the given input text in the same language.
        
        Args:
            input_text (str): The input text to process
            **kwargs: Additional parameters for the model
            
        Returns:
            Dict[str, Any]: Response containing generated text and metadata
        """
        try:
            input_language = language
            
            # Prepare system prompt to ensure output in the same language
            system_prompt = f"You are a helpful assistant. Please respond in {input_language} and keep your responses laconic."
            
            # Prepare the request parameters
            request_params = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_text}
                ],
                "options": {
                    "temperature": self.temperature
                }
            }
            
            # Add any additional kwargs to options
            if kwargs:
                request_params["options"].update(kwargs)
            
            # Generate response using Ollama (removed await)
            response = ollama.chat(**request_params)
            data = {
                "generated_text": response.message.content,
                "input_language": input_language,
                "model_name": self.model_name,
                "status": "success"
            }

            data = LLMResult(**data)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return {
                "generated_text": "",
                "input_language": None,
                "model_name": self.model_name,
                "status": "error",
                "error_message": str(e)
            }
    
    async def batch_generate(self, input_texts: list[str]) -> list[Dict[str, Any]]:
        """
        Process multiple input texts concurrently.
        
        Args:
            input_texts (list[str]): List of input texts to process
            
        Returns:
            list[Dict[str, Any]]: List of response dictionaries
        """
        tasks = [self.generate_response(text) for text in input_texts]
        return await asyncio.gather(*tasks)
    
    def validate_input(self, input_text: str) -> bool:
        """
        Validate the input text before processing.
        
        Args:
            input_text (str): The input text to validate
            
        Returns:
            bool: True if input is valid, False otherwise
        """
        if not input_text or not isinstance(input_text, str):
            return False
        if len(input_text.strip()) == 0:
            return False
        return True 