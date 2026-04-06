"""
Loads .env and exports all settings as typed constants.
Phase 3 — Week 8: MCP Foundations. Single source of truth for all configuration.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Suppress ChromaDB's internal telemetry (posthog version mismatch causes noisy warnings)
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

# LLM
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.environ.get("OPENAI_MODEL", "gpt-4o")

# Ports
MAIN_API_PORT: int = int(os.environ.get("MAIN_API_PORT", "8000"))
MCP_SERVER_PORT: int = int(os.environ.get("MCP_SERVER_PORT", "8001"))

# Data paths
DB_PATH: str = os.environ.get("DB_PATH", "./data/research.db")
PDF_DIR: str = os.environ.get("PDF_DIR", "./data/pdfs")
VECTORSTORE_DIR: str = os.environ.get("VECTORSTORE_DIR", "./data/vectorstore")

# Embedding model
EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Search defaults
DEFAULT_MAX_RESULTS: int = int(os.environ.get("DEFAULT_MAX_RESULTS", "20"))
DEFAULT_YEAR_FROM: int = int(os.environ.get("DEFAULT_YEAR_FROM", "2022"))

# Pre-filter settings
KEYWORD_FILTER_ENABLED: bool = os.environ.get("KEYWORD_FILTER_ENABLED", "true").lower() == "true"
MIN_ABSTRACT_LENGTH: int = int(os.environ.get("MIN_ABSTRACT_LENGTH", "50"))
