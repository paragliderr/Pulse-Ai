import os
from pathlib import Path
from dotenv import load_dotenv

# Same fix as the backend: load_dotenv() with no path only checks the
# current working directory, which breaks if this is launched from inside
# recommendation-engine/ instead of the repo root. Point it at the root .env
# explicitly, two levels up from this file.
_root_env = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=_root_env)

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")
DATABASE_URL = os.getenv("DATABASE_URL_PYTHON", "")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Ticker universe for this milestone — see Milestone 2 report, Section 3 (Dataset).
TICKER_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "TSLA", "META", "JPM", "V", "UNH",
    # extend to ~50 tickers across sectors
]

# Model hyperparameters (starting points — tune during Milestone 3 experiments)
LIGHTGCN_CONFIG = {
    "embedding_dim": 64,
    "n_layers": 3,
    "learning_rate": 1e-3,
}

SRGNN_CONFIG = {
    "hidden_size": 100,
    "n_gnn_steps": 1,
    "learning_rate": 1e-3,
}

SESSION_GAP_MINUTES = 30  # time-gap threshold for session segmentation
