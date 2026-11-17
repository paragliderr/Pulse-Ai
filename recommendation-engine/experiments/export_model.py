"""
Trains the SELECTED Pulse-Hybrid configuration (fixed-average fusion) on the real-ticker universe
(app/universe.py) and exports it for the API.

    cd recommendation-engine/experiments
    python export_model.py              # about 1 minute, writes ../app/model_store/pulse_model.pt

The checkpoint is small (about 0.4 MB). Commit it so teammates get the model without retraining.
The data is SIMULATED (see sim_data.py); the file's meta says so and /model/info reports it.
"""
import sys, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.universe import SIM_UNIVERSE, UNIVERSE, STORIES_PER_TICKER                       # noqa: E402
from sim_data import generate, split, cases_from, edges_from                              # noqa: E402
from models import PulseHybrid                                                             # noqa: E402
import train_eval as te                                                                    # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "app" / "model_store" / "pulse_model.pt"


def main(seed=0, epochs=25, patience=4):
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)
    torch.manual_seed(seed); np.random.seed(seed)

    ds = generate(seed=seed, universe=SIM_UNIVERSE)
    train, val, test = split(ds)
    tr_eu, tr_ei = [torch.from_numpy(x) for x in edges_from(train)]
    te_eu, te_ei = [torch.from_numpy(x) for x in edges_from(train, val)]
    warm = set(np.where(~ds.cold_users)[0].tolist())
    train_cases = [c for u, ss in train.items() for s in ss for c in cases_from({u: s})]
    val_cases, test_cases = cases_from(val, warm), cases_from(test)
    log(f"{ds.n_items} items, {len(train_cases)} training cases")

    model = PulseHybrid(ds.n_users, ds.n_items, ds.item_feat, use_gate=False)
    model = te.train_session_model(model, ds, train_cases, val_cases, epochs, 2e-3, patience, log,
                                   edges=(tr_eu, tr_ei), hybrid=True)

    scores = te.score_hybrid(model, test_cases, te_eu, te_ei, ds.n_users)
    metrics = te.split_metrics(scores, test_cases, ds.cold_users)
    log("held-out test (simulated data): " + str({k: round(v["NDCG@10"], 4) for k, v in metrics.items()}))

    model.eval()
    with torch.no_grad():
        _u, i_prop, _content = model.item_table(te_eu, te_ei)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "i_prop": i_prop.clone(),
        "content_state": model.content.state_dict(),
        "enc_state": model.enc.state_dict(),
        "steps": model.enc.steps,
        "item_feat": ds.item_feat,
        "meta": {
            "description": "Pulse-Hybrid, fixed-average fusion, trained on SIMULATED sessions over the real-ticker universe",
            "trained_on": "simulated",
            "seed": seed, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tickers": len(UNIVERSE), "stories_per_ticker": STORIES_PER_TICKER,
            "test_ndcg10_simulated": {k: round(v["NDCG@10"], 4) for k, v in metrics.items()},
            "serving_note": "app users are served through the session + content path (no long-term user vector)",
        },
    }, OUT)
    log(f"saved {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
