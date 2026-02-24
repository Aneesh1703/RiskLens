# GenAI integration layer for insider threat explainability
import os
from google import genai


def get_client(api_key: str = None, model_name: str = None):
    """Configure and return a Gemini client + model name."""
    if model_name is None:
        model_name = "gemini-2.5-flash"
    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY")
    if api_key is None:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    client = genai.Client(api_key=api_key)
    return client, model_name