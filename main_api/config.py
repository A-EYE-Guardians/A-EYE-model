import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Settings:
    host: str = os.getenv("MAIN_API_HOST", "0.0.0.0")
    port: int = int(os.getenv("MAIN_API_PORT", "9000"))
    langgraph_url: str = os.getenv("LANGGRAPH_URL", "http://127.0.0.1:3020/infer")
    langgraph_api_key: str = os.getenv("LANGGRAPH_API_KEY", "")
    http_timeout_s: int = int(os.getenv("HTTP_TIMEOUT_SECONDS", "20"))
    http_retries: int = int(os.getenv("HTTP_RETRIES", "2"))

settings = Settings()
