from fastapi import FastAPI
from app.api import recommend

app = FastAPI(title="Pulse AI — Recommendation Engine", version="0.1.0")

app.include_router(recommend.router)


@app.get("/health")
def health():
    return {"status": "ok"}
