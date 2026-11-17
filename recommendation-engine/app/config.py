import os
from pathlib import Path
from dotenv import load_dotenv

# Load the repo-root .env no matter which folder the service is started from.
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")
DATABASE_URL = os.getenv("DATABASE_URL_PYTHON", "")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

TICKER_UNIVERSE = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "JPM", "V", "UNH"]  # used by the stub only
SESSION_GAP_MINUTES = 30
