"""API contract tests. Work in both modes: they check whichever mode the engine is in."""
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["mode"] in ("model", "stub")


def test_generic_feed_has_items():
    r = client.post("/recommend", json={"user_id": "new", "watchlist": [], "top_k": 10})
    body = r.json()
    assert r.status_code == 200 and len(body["recommendations"]) > 0
    for it in body["recommendations"]:
        assert "score" in it and "explanation" in it
    if body["mode"] == "model":
        assert len({i["sector"] for i in body["recommendations"]}) >= 4     # market-wide feed spans sectors


def test_watchlist_and_session_feeds_differ_in_model_mode():
    base = client.post("/recommend", json={"user_id": "u", "watchlist": [], "top_k": 10}).json()
    if base["mode"] != "model":
        return
    wl = client.post("/recommend", json={"user_id": "u", "watchlist": ["XOM", "COP"], "top_k": 10}).json()
    ss = client.post("/recommend", json={"user_id": "u", "recent_items": ["NEE:0", "DUK:1", "SO:0"], "top_k": 10}).json()
    assert base["personalisation"] == "generic" and wl["personalisation"] == "watchlist" and ss["personalisation"] == "session"
    ids = lambda b: [i["item_id"] for i in b["recommendations"]]
    assert ids(base) != ids(wl) != ids(ss)
    assert [i["sector"] for i in ss["recommendations"]].count("Utilities") >= 3     # reading utilities surfaces utilities


def test_model_mode_explanation_adds_up_and_is_labelled():
    body = client.post("/recommend", json={"user_id": "u", "recent_items": ["AAPL:1", "MSFT:0"], "top_k": 5}).json()
    if body["mode"] != "model":
        return
    for it in body["recommendations"]:
        ex = it["explanation"]
        assert abs(sum(ex["parts"].values()) - ex["raw_score"]) < 0.02       # exact decomposition (rounding only)
        assert it["source"] in ("sample", "live") and it["headline"]
        assert 0 < it["score"] < 1
    assert not ({"AAPL:1", "MSFT:0"} & {i["item_id"] for i in body["recommendations"]})   # never recommends what was just read


def test_bad_input_is_ignored_not_fatal():
    r = client.post("/recommend", json={"user_id": "u", "recent_items": ["nope", "ZZZ:99", "NVDA:2"], "top_k": 5})
    assert r.status_code == 200
