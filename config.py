import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env into os.environ
load_dotenv()


class Config:
    PRJ_ROOT = Path(__file__).parent
    MODEL_ROOT = os.path.join(PRJ_ROOT, "trained_model")
    USER_FILES = os.getenv("USER_FILE_PATH", os.path.join(PRJ_ROOT, "user_files"))

    LOG_ROOT = os.path.join(USER_FILES, "logs")
    CACHE_ROOT = os.path.join(USER_FILES, "cache")
    DATA_ROOT = os.path.join(USER_FILES, "data")
    DB_ROOT = os.path.join(USER_FILES, "database")
    OUTPUT_ROOT = os.path.join(USER_FILES, "output")
    INPUT_ROOT = os.path.join(USER_FILES, "input")
    CONFIG_ROOT = os.path.join(USER_FILES, "config")

    DOCUMENT_REGISTRY_PATH = os.path.join(USER_FILES, "registry")
    
    OPEN_ROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPEN_ROUTER_API_BASE: str = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    HOST_AGENT_PORT: int = os.getenv("DB_URL", 8080)
    POST_DESIGN_AGENT_PORT: int = os.getenv("POST_DESIGN_AGENT_PORT", 10000)
    MATH_MCP_SERVER_PORT: int = os.getenv("MATH_MCP_SERVER_PORT", 10001)
    
    OPENINFERENCE_DISABLED: bool = os.getenv("OPENINFERENCE_DISABLED", False)
    DEBUG: bool = os.getenv("DEBUG", False)
    
    def __init__(self):
        # Create directories if they don't exist
        for path in [
            self.USER_FILES,
            self.LOG_ROOT,
            self.CACHE_ROOT,
            self.DATA_ROOT,
            self.DB_ROOT,
            self.OUTPUT_ROOT,
            self.INPUT_ROOT,
            self.CONFIG_ROOT,
            self.DOCUMENT_REGISTRY_PATH,
        ]:
            os.makedirs(path, exist_ok=True)


# You can create a global instance if you like
config = Config()

from langfuse import Langfuse

langfuse = Langfuse(
  secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
  public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
  host=os.getenv("LANGFUSE_HOST"),
)
# # Verify connection
# if langfuse.auth_check():
#     print("Langfuse client is authenticated and ready!")
# else:
#     print("Authentication failed. Please check your credentials and host.")
