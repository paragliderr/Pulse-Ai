"""
Explainability layer — SHAP/LIME attribution on the ranking model's output.

STATUS: `explain_stub` returns plausible-looking dummy feature contributions
so the API response shape is stable for frontend/backend integration.
Replace with a real `shap.Explainer` (or `lime`) call once the fusion model
in models/fusion.py is trained — see Milestone 2 report, Section 6.
"""

import random
from typing import Dict


CANDIDATE_FEATURES = [
    "sentiment_spike",
    "sector_momentum",
    "similarity_to_watchlist",
    "price_momentum",
    "persona_match",
]


def explain_stub(item_id: str) -> Dict[str, float]:
    chosen = random.sample(CANDIDATE_FEATURES, k=random.randint(2, 3))
    return {feature: round(random.uniform(0.1, 0.6), 2) for feature in chosen}


# TODO: real implementation once fusion.py is trained
# import shap
# def explain(model, background_data, instance):
#     explainer = shap.Explainer(model, background_data)
#     shap_values = explainer(instance)
#     return dict(zip(feature_names, shap_values.values[0]))
