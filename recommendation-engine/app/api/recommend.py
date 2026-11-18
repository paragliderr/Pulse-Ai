from typing import List
from fastapi import APIRouter
from pydantic import BaseModel
from app.models.fusion import rank

router = APIRouter()


class RecommendRequest(BaseModel):
    user_id: str = "anon"
    watchlist: List[str] = []
    recent_items: List[str] = []   # "TICKER:slot", oldest first
    top_k: int = 10


@router.post("/recommend")
def recommend(req: RecommendRequest):
    return rank(req.watchlist, req.recent_items, req.top_k)
