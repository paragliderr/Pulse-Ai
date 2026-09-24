from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any

from app.models.fusion import rank_for_user

router = APIRouter()


class RecommendRequest(BaseModel):
    user_id: str
    watchlist: List[str] = []
    top_k: int = 20


class RecommendationItem(BaseModel):
    item_id: str
    item_type: str  # "ticker" | "news"
    score: float
    explanation: Dict[str, Any]  # feature -> contribution, from SHAP/LIME


class RecommendResponse(BaseModel):
    recommendations: List[RecommendationItem]


@router.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    """
    Backend calls this with a user_id and their current watchlist.
    If watchlist is empty, this should fall back to a general/cold-start
    feed (SR-GNN branch handles this case) rather than personalized ranking.
    """
    results = rank_for_user(user_id=req.user_id, watchlist=req.watchlist, top_k=req.top_k)
    return RecommendResponse(recommendations=results)
