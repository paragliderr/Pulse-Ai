from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_recommend_with_watchlist():
    resp = client.post("/recommend", json={"user_id": "u1", "watchlist": ["AAPL", "MSFT"], "top_k": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert "recommendations" in body
    assert len(body["recommendations"]) <= 5
    for item in body["recommendations"]:
        assert "score" in item and "explanation" in item


def test_recommend_cold_start_no_watchlist():
    resp = client.post("/recommend", json={"user_id": "new_user", "watchlist": [], "top_k": 5})
    assert resp.status_code == 200
    assert len(resp.json()["recommendations"]) > 0
