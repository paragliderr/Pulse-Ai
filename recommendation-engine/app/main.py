from fastapi import FastAPI
from app.api import recommend

app = FastAPI(title="Pulse AI - Recommendation Engine")
app.include_router(recommend.router)


@app.get("/health")
def health():
    return {"status": "ok", "mode": "simple"}


@app.get("/model/info")
def model_info():
    return {
        "name": "pulse-content-ranker",
        "type": "content-based (sector + ticker + recency)",
        "trained_on": "none - rule-based, no training data",
        "tickers": 40,
        "catalogue": "sample (placeholder headlines)",
    }
