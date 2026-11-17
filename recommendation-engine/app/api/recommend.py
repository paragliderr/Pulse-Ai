from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from app.models.fusion import rank_for_user

router = APIRouter()


class RecommendRequest(BaseModel):
    user_id: str
    watchlist: List[str] = []
    recent_items: List[str] = []   # e.g. ["NVDA:2", "AMD:0"] - the user's latest clicks, oldest first
    top_k: int = 20


class RecommendationItem(BaseModel):
    item_id: str
    item_type: str                 # "news" (model mode) | "ticker" (stub mode)
    score: float
    explanation: Dict[str, Any]
    ticker: Optional[str] = None
    sector: Optional[str] = None
    market_cap_tier: Optional[str] = None
    headline: Optional[str] = None
    source: Optional[str] = None   # "live" | "sample"
    url: Optional[str] = None


class RecommendResponse(BaseModel):
    recommendations: List[RecommendationItem]
    mode: str                      # "model" | "stub"
    personalisation: str           # "session" | "watchlist" | "generic" | "none"
    catalogue_mode: str            # "live" | "sample" | "none"


@router.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    res = rank_for_user(req.user_id, req.watchlist, req.top_k, req.recent_items)
    return RecommendResponse(recommendations=res["items"], mode=res["mode"],
                             personalisation=res["personalisation"], catalogue_mode=res["catalogue_mode"])
