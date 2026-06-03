import os
import socket

# Force IPv4 resolution to prevent connection hangs on hosts with broken IPv6
old_getaddrinfo = socket.getaddrinfo
def new_getaddrinfo(*args, **kwargs):
    responses = old_getaddrinfo(*args, **kwargs)
    return [res for res in responses if res[0] == socket.AF_INET]
socket.getaddrinfo = new_getaddrinfo

from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI

# hw2/src/config.py

# LLM Provider Configuration
# Supported providers: "gemini", "openai", "anthropic", "groq"
LLM_PROVIDER = "groq"  # Defaulting to groq to avoid Gemini rate limits

# RAG Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_K = 3

def validate_api_key():
    """Validates that the appropriate API key is set in the environment."""
    if LLM_PROVIDER == "gemini" and "GOOGLE_API_KEY" not in os.environ and "GEMINI_API_KEY" not in os.environ:
        return False, "Google/Gemini API key is not configured."
    elif LLM_PROVIDER == "openai" and "OPENAI_API_KEY" not in os.environ:
        return False, "OpenAI API key is not configured."
    elif LLM_PROVIDER == "anthropic" and "ANTHROPIC_API_KEY" not in os.environ:
        return False, "Anthropic API key is not configured."
    elif LLM_PROVIDER == "groq" and "GROQ_API_KEY" not in os.environ:
        return False, "Groq API key is not configured."
    return True, ""

def get_llm():
    """Instantiates and returns the appropriate LLM based on LLM_PROVIDER."""
    if LLM_PROVIDER == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "dummy"
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, api_key=api_key)
    elif LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        api_key = os.environ.get("OPENAI_API_KEY") or "dummy"
        return ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
    elif LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY") or "dummy"
        return ChatAnthropic(model="claude-3-haiku-20240307", temperature=0, api_key=api_key)
    elif LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        import httpx
        api_key = os.environ.get("GROQ_API_KEY") or "dummy"
        http_client = httpx.Client(transport=httpx.HTTPTransport(local_address="0.0.0.0"))
        return ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=api_key, http_client=http_client)
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")
