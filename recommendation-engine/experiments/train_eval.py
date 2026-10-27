"""
Pulse AI - Milestone 3 experiment runner.

    python train_eval.py                 # 3 seeds, full run (a few minutes per seed on CPU)
    python train_eval.py --quick         # 1 seed, tiny run, just to check the setup works
    python train_eval.py --seeds 0 1     # choose seeds

Outputs (in ./results): results.json, results_table.csv, explanation_example.json
"""

import argparse, json, time, copy
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from sim_data import generate, split, cases_from, edges_from
from models import BPRMF, LightGCN, SRGNN, PulseHybrid, session_batch

torch.set_num_threads(1)
HIST_NORM = np.log1p(60.0)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def rank_metrics(scores, targets):
    tgt = scores[np.arange(len(targets)), targets]
    ranks = (scores >= tgt[:, None]).sum(1)
    out = {}
    for k in (10, 20):
        out[f"HR@{k}"] = float((ranks <= k).mean())
        out[f"NDCG@{k}"] = float(np.where(ranks <= k, 1 / np.log2(ranks + 1), 0).mean())
    out["MRR@20"] = float(np.where(ranks <= 20, 1 / ranks, 0).mean())
    return out


def split_metrics(scores, cases, cold_mask):
    users = np.array([c[0] for c in cases]); targets = np.array([c[2] for c in cases])
    is_cold = cold_mask[users]
    res = {"overall": rank_metrics(scores, targets),
           "warm": rank_metrics(scores[~is_cold], targets[~is_cold]),
           "cold": rank_metrics(scores[is_cold], targets[is_cold])}
    return res


# --------------------------------------------------------------------------
# scoring helpers (one per model family)
# --------------------------------------------------------------------------
def hist_features(users, eu, n_users):
    deg = torch.bincount(eu, minlength=n_users).float() if eu.numel() else torch.zeros(n_users)
    d = deg[users]
    return (d > 0).float(), torch.clamp(torch.log1p(d) / HIST_NORM, max=1.0)


@torch.no_grad()
def score_mf(model, cases, eu, ei):
    model.eval()
    U, I = model.all_embeddings(eu, ei)
    users = torch.tensor([c[0] for c in cases])
    return (U[users] @ I.T).numpy()


@torch.no_grad()
def score_srgnn(model, cases, bs=512):
    model.eval()
    out = []
    for s in range(0, len(cases), bs):
        chunk = cases[s:s + bs]
        out.append(model(session_batch([c[1] for c in chunk])).numpy())
    return np.concatenate(out)


@torch.no_grad()
def score_hybrid(model, cases, eu, ei, n_users, bs=512, parts=False):
    model.eval()
    out = []
    for s in range(0, len(cases), bs):
        chunk = cases[s:s + bs]
        users = torch.tensor([c[0] for c in chunk])
        hm, conf = hist_features(users, eu, n_users)
        out.append(model(users, session_batch([c[1] for c in chunk]), eu, ei, hm, conf).numpy())
    return np.concatenate(out)


# --------------------------------------------------------------------------
# training loops
# --------------------------------------------------------------------------
def train_bpr(model, ds, tr_eu, tr_ei, val_cases, epochs, lr, patience, log):
    n_u, n_i = ds.n_users, ds.n_items
    pos = torch.zeros(n_u, n_i, dtype=torch.bool); pos[tr_eu, tr_ei] = True
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, bad = -1, None, 0
    val_targets = np.array([c[2] for c in val_cases])
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(tr_eu))
        for s in range(0, len(perm), 1024):
            idx = perm[s:s + 1024]
            u, i = tr_eu[idx], tr_ei[idx]
            j = torch.randint(0, n_i, (len(u),))
            j = torch.where(pos[u, j], torch.randint(0, n_i, (len(u),)), j)
            U, I = model.all_embeddings(tr_eu, tr_ei)
            x = (U[u] * I[i]).sum(1) - (U[u] * I[j]).sum(1)
            reg = (model.user(u).pow(2).sum() + model.item(i).pow(2).sum() + model.item(j).pow(2).sum()) / (2 * len(u))
            loss = -F.logsigmoid(x).mean() + 1e-4 * reg
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 2 == 1:
            m = rank_metrics(score_mf(model, val_cases, tr_eu, tr_ei), val_targets)["NDCG@20"]
            if m > best:
                best, bad, best_state = m, 0, copy.deepcopy(model.state_dict())
            else:
                bad += 1
                if bad >= patience: break
    log(f"    best val NDCG@20 = {best:.4f}")
    model.load_state_dict(best_state)
    return model


def train_session_model(model, ds, train_cases, val_cases, epochs, lr, patience, log,
                        edges=None, hybrid=False, cold_aug=0.2, bs=256):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    n_u, n_i = ds.n_users, ds.n_items
    val_targets = np.array([c[2] for c in val_cases])
    if hybrid and model.use_graph:
        eu, ei = edges
        edge_keys = eu * n_i + ei
    best, best_state, bad = -1, None, 0
    for ep in range(epochs):
        model.train()
        order = np.random.permutation(len(train_cases))
        for s in range(0, len(order), bs):
            chunk = [train_cases[k] for k in order[s:s + bs]]
            targets = torch.tensor([c[2] for c in chunk])
            batch = session_batch([c[1] for c in chunk])
            if not hybrid:
                loss = F.cross_entropy(model(batch), targets)
            else:
                users = torch.tensor([c[0] for c in chunk])
                if model.use_graph:
                    # leave-target-edge-out: the graph must not contain the edge it is asked to predict
                    keep = ~torch.isin(edge_keys, users * n_i + targets)
                    eu_m, ei_m = eu[keep], ei[keep]
                else:
                    eu_m = ei_m = torch.zeros(0, dtype=torch.long)
                hm, conf = hist_features(users, eu_m, n_u)
                drop = torch.rand(len(users)) < cold_aug         # simulate brand-new users
                hm = hm * (~drop).float(); conf = conf * (~drop).float()
                loss = F.cross_entropy(model(users, batch, eu_m, ei_m, hm, conf), targets)
            opt.zero_grad(); loss.backward(); opt.step()
        if hybrid:
            sc = score_hybrid(model, val_cases, edges[0], edges[1], n_u)
        else:
            sc = score_srgnn(model, val_cases)
        m = rank_metrics(sc, val_targets)["NDCG@20"]
        if m > best:
            best, bad, best_state = m, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience: break
    log(f"    best val NDCG@20 = {best:.4f}  (stopped at epoch {ep + 1})")
    model.load_state_dict(best_state)
    return model


# --------------------------------------------------------------------------
# fairness-aware re-ranking + exposure statistics
# --------------------------------------------------------------------------
def exposure_stats(scores, tiers, k=10):
    topk = np.argsort(-scores, axis=1)[:, :k]
    t = tiers[topk]
    return {"coverage@10": float(len(np.unique(topk)) / scores.shape[1]),
            "mega_share@10": float((t == 0).mean()),
            "large_share@10": float((t == 1).mean()),
            "mid_share@10": float((t == 2).mean())}


def rerank(scores, tiers, lam, k=10, pool=50):
    """Greedy diversity re-ranking: every already-chosen item of the same
    market-cap tier lowers a candidate's (z-scored) relevance by `lam`."""
    out = np.zeros((scores.shape[0], k), dtype=np.int64)
    for r in range(scores.shape[0]):
        cand = np.argsort(-scores[r])[:pool]
        z = (scores[r, cand] - scores[r, cand].mean()) / (scores[r, cand].std() + 1e-8)
        chosen, counts = [], np.zeros(3)
        for _ in range(k):
            adj = z - lam * counts[tiers[cand]]
            adj[[cand.tolist().index(c) for c in chosen]] = -1e9
            p = int(np.argmax(adj))
            chosen.append(int(cand[p])); counts[tiers[cand[p]]] += 1
        out[r] = chosen
    return out


def topk_metrics(topk, targets, k=10):
    hit = topk[:, :k] == targets[:, None]
    ranks = np.where(hit.any(1), hit.argmax(1) + 1, 10 ** 6)
    return {"HR@10": float((ranks <= k).mean()),
            "NDCG@10": float(np.where(ranks <= k, 1 / np.log2(ranks + 1), 0).mean())}


# --------------------------------------------------------------------------
# explanation (exact additive decomposition of the hybrid score)
# --------------------------------------------------------------------------
@torch.no_grad()
def explain_case(model, ds, case, eu, ei, n_users, top=3):
    model.eval()
    u, prefix, target = case
    users = torch.tensor([u]); batch = session_batch([prefix])
    hm, conf = hist_features(users, eu, n_users)
    scores, p = model(users, batch, eu, ei, hm, conf, parts=True)
    top_items = torch.topk(scores[0], top).indices.tolist()

    def forward_with_feat(feat):
        old = model.feat.clone(); model.feat.copy_(feat)
        sc = model(users, batch, eu, ei, hm, conf)[0]
        model.feat.copy_(old)
        return sc

    group_scores = {}
    for name, cols in ds.feat_groups.items():
        f = model.feat.clone(); f[:, cols] = f[:, cols].mean(0, keepdim=True)
        group_scores[name] = forward_with_feat(f)

    rows = []
    for j in top_items:
        parts = {
            "long_term_collaborative": float(p["gu"][0] @ p["i_prop"][j]),
            "long_term_content": float(p["gu"][0] @ p["content"][j]),
            "session_collaborative": float(p["gs"][0] @ p["i_prop"][j]),
            "session_content": float(p["gs"][0] @ p["content"][j]),
        }
        rows.append({
            "item": j, "ticker": int(ds.item_ticker[j]), "score": float(scores[0, j]),
            "is_actual_next_click": bool(j == target),
            "decomposition": parts,
            "feature_group_effect": {n: float(scores[0, j] - s[j]) for n, s in group_scores.items()},
        })
    return {"user": int(u), "prefix_len": int(len(prefix)),
            "history_available": bool(hm.item() > 0),
            "long_term_share_of_user_vector": float(p["gu"].norm() / (p["gu"].norm() + p["gs"].norm() + 1e-8)), "top_recommendations": rows}


# --------------------------------------------------------------------------
# main experiment for one seed
# --------------------------------------------------------------------------
def run_seed(seed, quick, log):
    torch.manual_seed(seed); np.random.seed(seed)
    ds = generate(seed=seed)
    train, val, test = split(ds)
    cold = ds.cold_users
    tr_eu, tr_ei = [torch.from_numpy(x) for x in edges_from(train)]
    te_eu, te_ei = [torch.from_numpy(x) for x in edges_from(train, val)]
    warm_users = set(np.where(~cold)[0].tolist())

    train_cases = [c for u, ss in train.items() for s in ss for c in cases_from({u: s})]
    val_cases = cases_from(val, warm_users)
    test_cases = cases_from(test)
    test_targets = np.array([c[2] for c in test_cases])
    log(f"  users={ds.n_users} items={ds.n_items} train_cases={len(train_cases)} "
        f"val_cases={len(val_cases)} test_cases={len(test_cases)} cold_users={int(cold.sum())}")

    E_BPR, E_SESS, PAT = (6, 3, 2) if quick else (80, 25, 4)
    results, test_scores = {}, {}

    def record(name, scores):
        results[name] = split_metrics(scores, test_cases, cold)
        test_scores[name] = scores
        o = results[name]["overall"]; c = results[name]["cold"]
        log(f"    -> {name:34s} overall NDCG@10={o['NDCG@10']:.4f} HR@10={o['HR@10']:.4f} | cold NDCG@10={c['NDCG@10']:.4f}")

    # --- popularity
    pop = np.bincount(np.concatenate([s for ss in train.values() for s in ss]), minlength=ds.n_items).astype(float)
    record("Popularity", np.tile(pop, (len(test_cases), 1)))

    # --- BPR-MF and LightGCN (long-term only)
    for name, cls in (("BPR-MF", lambda: BPRMF(ds.n_users, ds.n_items)),
                      ("LightGCN", lambda: LightGCN(ds.n_users, ds.n_items))):
        log(f"  training {name}")
        m = train_bpr(cls(), ds, tr_eu, tr_ei, val_cases, E_BPR, 0.01, PAT, log)
        record(name, score_mf(m, test_cases, te_eu, te_ei))

    # --- SR-GNN (session only)
    log("  training SR-GNN")
    m = train_session_model(SRGNN(ds.n_items), ds, train_cases, val_cases, E_SESS, 2e-3, PAT, log)
    record("SR-GNN", score_srgnn(m, test_cases))

    # --- Pulse-Hybrid (proposed) and ablations
    variants = [
        ("Pulse-Hybrid (proposed)", dict()),
        ("  - gate (fixed 0.5 average)", dict(use_gate=False)),
        ("  - content features", dict(use_content=False)),
        ("  - session branch", dict(use_session=False)),
        ("  - graph branch", dict(use_graph=False)),
    ]
    if quick:
        variants = variants[:1]
    hybrid_main = None
    for name, flags in variants:
        log(f"  training {name.strip()}")
        m = PulseHybrid(ds.n_users, ds.n_items, ds.item_feat, **flags)
        m = train_session_model(m, ds, train_cases, val_cases, E_SESS, 2e-3, PAT, log,
                                edges=(tr_eu, tr_ei), hybrid=True)
        record(name, score_hybrid(m, test_cases, te_eu, te_ei, ds.n_users))
        if name.startswith("Pulse-Hybrid"):
            hybrid_main = m

    # --- exposure / fairness
    tiers = ds.item_tier
    target_tiers = tiers[test_targets]
    fairness = {"ground_truth": {"mega_share@10": float((target_tiers == 0).mean()),
                                 "large_share@10": float((target_tiers == 1).mean()),
                                 "mid_share@10": float((target_tiers == 2).mean())}}
    for name in ("Popularity", "LightGCN", "SR-GNN", "Pulse-Hybrid (proposed)"):
        fairness[name] = exposure_stats(test_scores[name], tiers)

    tradeoff = []
    for lam in (0.0, 0.25, 0.5, 1.0, 2.0):
        tk = rerank(test_scores["Pulse-Hybrid (proposed)"], tiers, lam)
        row = {"lambda": lam, **topk_metrics(tk, test_targets)}
        t = tiers[tk]
        row.update({"coverage@10": float(len(np.unique(tk)) / ds.n_items),
                    "mega_share@10": float((t == 0).mean()), "mid_share@10": float((t == 2).mean())})
        tradeoff.append(row)

    explanations = None
    if hybrid_main is not None:
        warm_case = next(c for c in test_cases if not cold[c[0]] and len(c[1]) >= 3)
        cold_case = next(c for c in test_cases if cold[c[0]] and len(c[1]) >= 3)
        explanations = {"warm_user": explain_case(hybrid_main, ds, warm_case, te_eu, te_ei, ds.n_users),
                        "cold_user": explain_case(hybrid_main, ds, cold_case, te_eu, te_ei, ds.n_users)}

        # effective share of the user vector that comes from long-term history (warm vs cold)
        gates = {"warm": [], "cold": []}
        with torch.no_grad():
            hybrid_main.eval()
            sample = test_cases[::7]
            for s in range(0, len(sample), 512):
                chunk = sample[s:s + 512]
                users = torch.tensor([c[0] for c in chunk])
                hm, conf = hist_features(users, te_eu, ds.n_users)
                _, p = hybrid_main(users, session_batch([c[1] for c in chunk]), te_eu, te_ei, hm, conf, parts=True)
                gm = (p["gu"].norm(dim=1) / (p["gu"].norm(dim=1) + p["gs"].norm(dim=1) + 1e-8)).numpy()
                for c, g in zip(chunk, gm):
                    gates["cold" if cold[c[0]] else "warm"].append(float(g))
        explanations["mean_long_term_share"] = {k: float(np.mean(v)) for k, v in gates.items()}

    return {"metrics": results, "fairness": fairness, "tradeoff": tradeoff, "explanations": explanations,
            "dataset": {"users": ds.n_users, "items": ds.n_items, "train_cases": len(train_cases),
                        "test_cases": len(test_cases), "cold_users": int(cold.sum())}}


def aggregate(per_seed):
    names = list(per_seed[0]["metrics"].keys())
    agg = {}
    for n in names:
        agg[n] = {}
        for grp in ("overall", "warm", "cold"):
            agg[n][grp] = {}
            for met in per_seed[0]["metrics"][n][grp]:
                vals = [r["metrics"][n][grp][met] for r in per_seed]
                agg[n][grp][met] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--quick", action="store_true", help="tiny run to verify the setup")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    seeds = [0] if args.quick else args.seeds

    Path(args.out).mkdir(exist_ok=True)
    t0 = time.time()
    def log(msg): print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    per_seed = []
    for sd in seeds:
        log(f"=== seed {sd} ===")
        per_seed.append(run_seed(sd, args.quick, log))
        json.dump({"per_seed": per_seed, "aggregate": aggregate(per_seed), "seeds": seeds[:len(per_seed)]},
                  open(Path(args.out) / "results.json", "w"), indent=2)

    agg = aggregate(per_seed)
    with open(Path(args.out) / "results_table.csv", "w") as f:
        f.write("model,group,HR@10,HR@20,NDCG@10,NDCG@20,MRR@20\n")
        for n, grp in agg.items():
            for g, mets in grp.items():
                f.write(f"{n.strip()},{g}," + ",".join(f"{mets[k]['mean']:.4f}" for k in
                        ("HR@10", "HR@20", "NDCG@10", "NDCG@20", "MRR@20")) + "\n")
    if per_seed[0]["explanations"]:
        json.dump(per_seed[0]["explanations"], open(Path(args.out) / "explanation_example.json", "w"), indent=2)
    print("\n=== SUMMARY (mean over seeds) ===")
    print(f"{'model':32s} {'group':8s} {'HR@10':>7s} {'NDCG@10':>8s} {'NDCG@20':>8s} {'MRR@20':>7s}")
    for n, grp in agg.items():
        for g in ("overall", "warm", "cold"):
            m = grp[g]
            print(f"{n:32s} {g:8s} {m['HR@10']['mean']:7.4f} {m['NDCG@10']['mean']:8.4f} "
                  f"{m['NDCG@20']['mean']:8.4f} {m['MRR@20']['mean']:7.4f}")
    log("done. results written to " + args.out)


if __name__ == "__main__":
    main()