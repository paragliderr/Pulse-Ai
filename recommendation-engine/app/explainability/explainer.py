"""Placeholder explanation used only when the trained model is not loaded (stub mode)."""
import random
from typing import Dict

CANDIDATE_FEATURES = ["sentiment_spike", "sector_momentum", "similarity_to_watchlist", "price_momentum", "persona_match"]


def explain_stub(item_id: str) -> Dict[str, float]:
    chosen = random.sample(CANDIDATE_FEATURES, k=random.randint(2, 3))
    return {f: round(random.uniform(0.1, 0.6), 2) for f in chosen}
