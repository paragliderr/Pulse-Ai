"""News ingestion via NewsAPI. See Milestone 2 report, Section 3."""

import requests
from app.config import NEWSAPI_KEY

NEWSAPI_URL = "https://newsapi.org/v2/everything"


def fetch_news_for_ticker(ticker: str, page_size: int = 20) -> list[dict]:
    """
    Returns a list of {headline, article_text, source, published_at, matched_ticker}.
    `article_text` is NewsAPI's `description` field — the full article body
    generally requires scraping or a paid tier, which is out of scope for now.
    """
    if not NEWSAPI_KEY:
        raise RuntimeError("NEWSAPI_KEY is not set — check your .env")

    params = {
        "q": ticker,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": NEWSAPI_KEY,
    }
    resp = requests.get(NEWSAPI_URL, params=params, timeout=10)
    resp.raise_for_status()
    articles = resp.json().get("articles", [])

    return [
        {
            "headline": a["title"],
            "article_text": a.get("description") or "",
            "source": a["source"]["name"],
            "published_at": a["publishedAt"],
            "matched_ticker": ticker,
        }
        for a in articles
    ]


if __name__ == "__main__":
    news = fetch_news_for_ticker("AAPL")
    print(f"Fetched {len(news)} articles")
    if news:
        print(news[0])
