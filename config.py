import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
LLAMA_API = os.getenv("LLAMA_API")
LANGSMITH_API = os.getenv("LANGSMITH_API")

# Model configuration
LLAMA_MODEL = "llama-70b-chat"  # or whatever model you want to use

# LangChain configuration
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API
os.environ["LANGCHAIN_PROJECT"] = "SOME_PROJECT_NAME"