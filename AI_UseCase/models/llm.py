import os
import sys
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from config.config import (
    GROQ_API_KEY,MODEL_NAME,OPENAI_API_KEY,OPENAI_MODEL,GOOGLE_API_KEY,GEMINI_MODEL)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


def get_chatgroq_model():
    """Initialize and return the Groq chat model"""
    try:
        # Initialize the Groq chat model with the API key
        return ChatGroq(
            api_key=GROQ_API_KEY,
            model=MODEL_NAME
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Groq model: {str(e)}")
def get_openai_model():
    try:
        return ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model=OPENAI_MODEL
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize OpenAI model: {str(e)}")
def get_gemini_model():
    try:
        return ChatGoogleGenerativeAI(
            google_api_key=GOOGLE_API_KEY,
            model=GEMINI_MODEL
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Gemini model: {str(e)}")
        
    