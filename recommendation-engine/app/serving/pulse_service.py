"""
Serves the exported Pulse-Hybrid model.

The app's users are not in the training data, so every request is served through the
model's session + content path (the same path new users take in the experiments, where
Pulse-Hybrid scored NDCG@10 0.083 for users with no history). The long-term user branch is
switched off (u = 0), and fusion is the selected fixed 0.5 average, i.e. z = 0.5 * session.

score(j) = z . (collaborative_j + content_j)  splits exactly into
  session x collaborative  +  session x content
and neutralising one feature group at a time gives that group's effect on the score.
"""
import numpy as np
import torch
import torch.nn as nn

from app.models.pulse_net import SessionEncoder, session_batch
from app.universe import UNIVERSE, TIERS, STORIES_PER_TICKER

GROUPS = {"sector": list(range(0, 8)), "market_cap": list(range(8, 11)), "sentiment": [11], "momentum": [12]}


class PulseService:
    def __init__(self, path: str):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        d, fdim = ck["i_prop"].shape[1], ck["item_feat"].shape[1]
        content = nn.Sequential(nn.Linear(fdim, d), nn.Tanh(), nn.Linear(d, d))
        content.load_state_dict(ck["content_state"])
        enc = SessionEncoder(d, ck["steps"])
        enc.load_state_dict(ck["enc_state"])
        self._setup(enc, content, ck["i_prop"].float(), np.asarray(ck["item_feat"], dtype=np.float32), ck["meta"])

    @classmethod
    def from_parts(cls, enc, content, i_prop, default_feat, meta=None):
        """Build a service from in-memory modules (used by the experiments)."""
        self = cls.__new__(cls)
        self._setup(enc, content, i_prop.float(), np.asarray(default_feat, dtype=np.float32), meta or {})
        return self

    def _setup(self, enc, content, i_prop, default_feat, meta):
        self.enc, self.content, self.i_prop = enc.eval(), content.eval(), i_prop
        self.default_feat, self.meta = default_feat, meta
        self.n_items = self.i_prop.shape[0]
        self.tier_of_item = np.array([UNIVERSE[i // STORIES_PER_TICKER][3] for i in range(self.n_items)])
        self.ticker_of_item = np.array([i // STORIES_PER_TICKER for i in range(self.n_items)])

    @torch.no_grad()
    def _scores(self, sessions, feat):
        """Mean over one or more sessions. Scores are linear in z, so the decomposition stays exact."""
        item_content = self.content(feat)
        item_final = self.i_prop + item_content
        zs = []
        for session in sessions:
            batch = session_batch([session])
            s = self.enc(item_final[batch["nodes"]], batch)
            zs.append(0.5 * s[0])                                # fixed-average fusion with u = 0
        z = torch.stack(zs).mean(0)
        return z @ item_final.T, z, item_content

    @torch.no_grad()
    def recommend(self, sessions, feat_np, top_k=20, exclude=(), tier_lambda=0.25, max_per_ticker=2, pool=60,
                  explain=True):
        feat = torch.as_tensor(feat_np, dtype=torch.float32)
        raw, z, item_content = self._scores(sessions, feat)
        shown = raw.clone()
        for i in set(exclude):
            shown[i] = -1e9
        cand = torch.topk(shown, min(pool, self.n_items)).indices.numpy()
        zs = raw[cand].numpy()
        zs = (zs - zs.mean()) / (zs.std() + 1e-8)

        # fairness-aware re-ranking: each already chosen item from the same market-cap tier (or ticker
        # beyond the cap) lowers a candidate's normalised score. lambda = 0 turns it off.
        chosen, tier_count, ticker_count = [], np.zeros(len(TIERS)), {}
        for _ in range(min(top_k, len(cand))):
            adj = zs - tier_lambda * tier_count[self.tier_of_item[cand]]
            for p, c in enumerate(cand):
                if c in chosen or ticker_count.get(self.ticker_of_item[c], 0) >= max_per_ticker:
                    adj[p] = -1e9
            p = int(np.argmax(adj))
            if adj[p] <= -1e8:
                break
            c = int(cand[p]); chosen.append(c)
            tier_count[self.tier_of_item[c]] += 1
            ticker_count[self.ticker_of_item[c]] = ticker_count.get(self.ticker_of_item[c], 0) + 1

        mu, sd = float(raw.mean()), float(raw.std() + 1e-8)
        group_scores = {}
        if explain:
            for name, cols in GROUPS.items():
                f = feat.clone(); f[:, cols] = f[:, cols].mean(0, keepdim=True)
                group_scores[name] = self._scores(sessions, f)[0]

        out = []
        for c in chosen:
            row = dict(index=c, raw=float(raw[c]), relevance=float(1 / (1 + np.exp(-(float(raw[c]) - mu) / sd))))
            if explain:
                row["parts"] = {"long_term_collaborative": 0.0, "long_term_content": 0.0,
                                "session_collaborative": float(z @ self.i_prop[c]),
                                "session_content": float(z @ item_content[c])}
                row["effects"] = {n: float(raw[c] - g[c]) for n, g in group_scores.items()}
            out.append(row)
        return out


def reasons_for(effects: dict, item: dict, how: str, top=2):
    """Plain-language reasons from the model's own feature-group effects (largest positive first)."""
    tier = item["market_cap_tier"].capitalize()
    wording = {
        "session":   (f"Fits the {item['sector']} stories in your recent reading", f"{tier}-cap, similar to your recent reading"),
        "watchlist": (f"{item['sector']}, like the stocks you follow", f"{tier}-cap, like the stocks you follow"),
        "generic":   (f"Active {item['sector']} story", f"{tier}-cap company"),
    }[how]
    texts = []
    for name, eff in sorted(effects.items(), key=lambda kv: -kv[1]):
        if eff <= 0.05 or len(texts) >= top:
            continue
        if name == "sector":
            texts.append(wording[0])
        elif name == "market_cap":
            texts.append(wording[1])
        elif name == "sentiment":
            sv = item["sentiment"]
            texts.append(("Positive" if sv > 0.15 else "Negative" if sv < -0.15 else "Neutral") + " news tone")
        elif name == "momentum":
            m = item["momentum"]
            texts.append("Recent price move up" if m > 0.15 else "Recent price move down" if m < -0.15 else "Steady recent price")
    return texts
