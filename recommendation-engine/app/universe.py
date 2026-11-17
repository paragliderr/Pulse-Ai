"""
The fixed ticker universe Pulse AI serves: 40 tickers, 5 per sector.

Each ticker owns STORIES_PER_TICKER "slots". Slot k of a ticker holds the k-th
latest headline for that company (slot 0 = newest). The recommender ranks these
(ticker, slot) items, so the model always sees a fixed catalogue of
40 x 10 = 400 items, while the headlines inside it refresh over time.

Sector and market-cap tier are APPROXIMATE labels (caps change). Edit freely,
then re-run  experiments/export_model.py  so the model is retrained on them.
"""

SECTORS = ["Technology", "Healthcare", "Financials", "Energy",
           "Consumer", "Industrials", "Utilities", "Materials"]
TIERS = ["mega", "large", "mid"]
STORIES_PER_TICKER = 10

# (symbol, company name for news search, sector index, tier index)
UNIVERSE = [
    ("AAPL", "Apple", 0, 0), ("MSFT", "Microsoft", 0, 0), ("NVDA", "Nvidia", 0, 0),
    ("AMD", "Advanced Micro Devices", 0, 1), ("ROKU", "Roku", 0, 2),

    ("LLY", "Eli Lilly", 1, 0), ("JNJ", "Johnson & Johnson", 1, 0), ("PFE", "Pfizer", 1, 1),
    ("GILD", "Gilead Sciences", 1, 1), ("MRNA", "Moderna", 1, 2),

    ("JPM", "JPMorgan", 2, 0), ("BAC", "Bank of America", 2, 1), ("GS", "Goldman Sachs", 2, 1),
    ("SCHW", "Charles Schwab", 2, 1), ("ALLY", "Ally Financial", 2, 2),

    ("XOM", "Exxon Mobil", 3, 0), ("CVX", "Chevron", 3, 1), ("COP", "ConocoPhillips", 3, 1),
    ("OXY", "Occidental Petroleum", 3, 1), ("APA", "APA Corporation", 3, 2),

    ("AMZN", "Amazon", 4, 0), ("WMT", "Walmart", 4, 0), ("NKE", "Nike", 4, 1),
    ("SBUX", "Starbucks", 4, 1), ("ETSY", "Etsy", 4, 2),

    ("GE", "GE Aerospace", 5, 0), ("CAT", "Caterpillar", 5, 1), ("UPS", "United Parcel Service", 5, 1),
    ("DAL", "Delta Air Lines", 5, 1), ("AAL", "American Airlines", 5, 2),

    ("NEE", "NextEra Energy", 6, 1), ("DUK", "Duke Energy", 6, 1), ("SO", "Southern Company", 6, 1),
    ("AEP", "American Electric Power", 6, 1), ("PNW", "Pinnacle West", 6, 2),

    ("LIN", "Linde", 7, 1), ("SHW", "Sherwin-Williams", 7, 1), ("FCX", "Freeport-McMoRan", 7, 1),
    ("NEM", "Newmont", 7, 1), ("CLF", "Cleveland-Cliffs", 7, 2),
]

SYMBOLS = [u[0] for u in UNIVERSE]
SYMBOL_INDEX = {s: i for i, s in enumerate(SYMBOLS)}
# what the simulator needs: (symbol, sector, tier)
SIM_UNIVERSE = [(u[0], u[2], u[3]) for u in UNIVERSE]
N_ITEMS = len(UNIVERSE) * STORIES_PER_TICKER


def item_id(symbol: str, slot: int) -> str:
    return f"{symbol}:{slot}"


def parse_item_id(text: str):
    """'NVDA:3' -> index; 'NVDA' -> that ticker's newest story. Returns None if unknown."""
    if not isinstance(text, str) or not text.strip():
        return None
    sym, _, slot = text.strip().upper().partition(":")
    if sym not in SYMBOL_INDEX:
        return None
    try:
        k = int(slot) if slot else 0
    except ValueError:
        return None
    if not 0 <= k < STORIES_PER_TICKER:
        return None
    return SYMBOL_INDEX[sym] * STORIES_PER_TICKER + k
