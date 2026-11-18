"""
News text preprocessing: cleaning + sentiment scoring.
See Milestone 2 report, Section 4 (Preprocessing Techniques).
"""

import re
from functools import lru_cache

TICKER_TOKEN_RE = re.compile(r"\$[A-Z]{1,5}\b")


def clean_headline(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


@lru_cache(maxsize=1)
def _get_sentiment_pipeline():
    # Loaded lazily so importing this module doesn't require downloading
    # the model until it's actually needed.
    from transformers import pipeline
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")


def score_sentiment(text: str) -> dict:
    """
    Returns {"label": "POSITIVE"|"NEGATIVE", "score": float}.
    NOTE: this is a general-purpose sentiment model as a starting point.
    A finance-tuned model (e.g. FinBERT) is a natural upgrade once the
    baseline pipeline is validated end-to-end.
    """
    pipe = _get_sentiment_pipeline()
    result = pipe(text[:512])[0]  # truncate to model's max input length
    return result


def deduplicate_headlines(articles: list[dict], similarity_threshold: float = 0.9) -> list[dict]:
    """
    Basic near-duplicate filtering on cleaned headlines using exact-match
    after cleaning as a first pass. TODO: replace with embedding-based
    similarity (e.g. sentence-transformers cosine similarity) for real
    near-duplicate detection across differently-worded headlines.
    """
    seen = set()
    deduped = []
    for article in articles:
        key = clean_headline(article["headline"])
        if key not in seen:
            seen.add(key)
            deduped.append(article)
    return deduped
