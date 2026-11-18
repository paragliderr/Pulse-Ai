from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    assert client.get("/health").status_code == 200


def test_session_beats_watchlist_and_hides_read():
    r = client.post("/recommend", json={"watchlist": ["XOM"], "recent_items": ["NEE:0"], "top_k": 5}).json()
    assert r["personalisation"] == "session"
    assert all(i["item_id"] != "NEE:0" for i in r["recommendations"])
    assert r["recommendations"][0]["sector"] == "Utilities"


def test_generic():
    r = client.post("/recommend", json={}).json()
    assert r["personalisation"] == "generic" and len(r["recommendations"]) == 10
