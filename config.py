import os
import secrets
from pathlib import Path
from dotenv import load_dotenv

# Locate base directory
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from .env if it exists
dotenv_path = BASE_DIR / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path)
else:
    load_dotenv()

# App Directories
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", str(BASE_DIR / "chroma_db"))
SANDBOX_DIR = str(BASE_DIR / "sandbox")

# Create standard directories if they don't exist
os.makedirs(CHROMA_DB_DIR, exist_ok=True)
os.makedirs(SANDBOX_DIR, exist_ok=True)

# LLM Configurations
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Databases
MONGODB_URI = os.getenv("MONGODB_URI", "")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "ai_github_assistant")

# GitHub Configuration
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

# Streamlit config
STREAMLIT_PORT = int(os.getenv("STREAMLIT_SERVER_PORT", "8501"))

# Encryption/Security Key (to secure GitHub tokens stored in MongoDB)
# We generate a consistent key or fallback to a dynamic one per run
SECRET_KEY_PATH = BASE_DIR / ".secret_key"
if SECRET_KEY_PATH.exists():
    with open(SECRET_KEY_PATH, "rb") as f:
        ENCRYPTION_KEY = f.read()
else:
    # Use a fixed fallback or create one
    ENCRYPTION_KEY = secrets.token_bytes(32)
    try:
        with open(SECRET_KEY_PATH, "wb") as f:
            f.write(ENCRYPTION_KEY)
    except Exception:
        pass

def is_gemini_available() -> bool:
    """Check if Gemini API Key is configured."""
    return bool(GEMINI_API_KEY)

def is_groq_available() -> bool:
    """Check if Groq API Key is configured."""
    return bool(GROQ_API_KEY)

def is_mongodb_configured() -> bool:
    """Check if MongoDB connection is specified."""
    return bool(MONGODB_URI)

def get_status_summary() -> dict:
    """Returns a dictionary showing configuration status of all core components."""
    return {
        "gemini": is_gemini_available(),
        "groq": is_groq_available(),
        "mongodb": is_mongodb_configured(),
        "github": bool(GITHUB_TOKEN),
        "chroma_dir": CHROMA_DB_DIR,
    }
