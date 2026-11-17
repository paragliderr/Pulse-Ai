from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api import recommend
from app.serving.registry import status


@asynccontextmanager
async def lifespan(_app):
    status()          # load the model once at startup so the first request is fast
    yield


app = FastAPI(title="Pulse AI - Recommendation Engine", version="0.2.0", lifespan=lifespan)
app.include_router(recommend.router)


@app.get("/health")
def health():
    st = status()
    return {"status": "ok", "mode": st["mode"], "detail": st["reason"]}


@app.get("/model/info")
def model_info():
    """What the served model is, and what it was trained on (so nobody mistakes it for a live-data model)."""
    return status()
