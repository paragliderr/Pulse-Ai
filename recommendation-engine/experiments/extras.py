"""
Pulse AI - remaining model experiments (explanations, drift, robustness, privacy, compression, personas).

    python extras.py                       # everything (a few minutes)
    python extras.py --only explain drift  # pick experiments
    python extras.py --quick               # tiny run, just to check the setup works

All experiments use the selected model (fixed-average Pulse-Hybrid) on the real-ticker universe with
SIMULATED sessions (see sim_data.py). Results are written to results/extras.json.
"""
import argparse, copy, io, itertools, json, time
from math import factorial
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.universe import SIM_UNIVERSE, UNIVERSE, SECTORS, TIERS, STORIES_PER_TICKER, SYMBOL_INDEX   # noqa: E402
from app.serving.pulse_service import PulseService, GROUPS                                         # noqa: E402
from sim_data import generate, split, cases_from, edges_from, simulate_sessions                   # noqa: E402
from models import PulseHybrid, session_batch                                                      # noqa: E402
import train_eval as te                                                                            # noqa: E402

torch.set_num_threads(1)
OUT = Path("results")
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def f4(x):
    return f"{x:.4f}"


# --------------------------------------------------------------------------
# shared setup: data, trained model, serving-path helpers
# --------------------------------------------------------------------------
class Ctx:
    pass


def build_ctx(quick):
    c = Ctx()
    torch.manual_seed(0); np.random.seed(0)
    c.ds = ds = generate(seed=0, universe=SIM_UNIVERSE)
    c.train, c.val, c.test = split(ds)
    c.tr_eu, c.tr_ei = [torch.from_numpy(x) for x in edges_from(c.train)]
    c.te_eu, c.te_ei = [torch.from_numpy(x) for x in edges_from(c.train, c.val)]
    warm = set(np.where(~ds.cold_users)[0].tolist())
    c.train_cases = [x for u, ss in c.train.items() for s in ss for x in cases_from({u: s})]
    c.val_cases, c.test_cases = cases_from(c.val, warm), cases_from(c.test)

    OUT.mkdir(exist_ok=True)
    cache = OUT / ("_model_quick.pt" if quick else "_model.pt")
    model = PulseHybrid(ds.n_users, ds.n_items, ds.item_feat, use_gate=False)
    if cache.exists():
        model.load_state_dict(torch.load(cache)); log(f"loaded cached model {cache.name}")
    else:
        log("training the selected model (cached afterwards)")
        model = te.train_session_model(model, ds, c.train_cases, c.val_cases, 3 if quick else 25, 2e-3,
                                       2 if quick else 4, log, edges=(c.tr_eu, c.tr_ei), hybrid=True)
        torch.save(model.state_dict(), cache)
    model.eval()
    c.model = model
    with torch.no_grad():
        _u, c.i_prop, _ = model.item_table(c.te_eu, c.te_ei)
    c.feat = torch.as_tensor(ds.item_feat)
    c.svc = PulseService.from_parts(model.enc, model.content, c.i_prop, ds.item_feat)
    c.tier = np.array([UNIVERSE[i // STORIES_PER_TICKER][3] for i in range(ds.n_items)])
    c.prefixes = [x[1] for x in c.test_cases]
    c.targets = np.array([x[2] for x in c.test_cases])
    return c


@torch.no_grad()
def serve_scores(enc, content, i_prop, feat, prefixes, bs=512):
    """Scores on the deployed path: session + content, no long-term user vector, fixed 0.5 fusion."""
    item_final = i_prop + content(feat)
    out = []
    for s in range(0, len(prefixes), bs):
        b = session_batch(prefixes[s:s + bs])
        out.append(((0.5 * enc(item_final[b["nodes"]], b)) @ item_final.T).numpy())
    return np.concatenate(out)


def serve_scores_grad(enc, content, i_prop, feat, prefixes):
    item_final = i_prop + content(feat)
    b = session_batch(prefixes)
    return (0.5 * enc(item_final[b["nodes"]], b)) @ item_final.T


# --------------------------------------------------------------------------
# 1. explanation validation: built-in effects vs exact Shapley vs LIME-style surrogate
# --------------------------------------------------------------------------
def exp_explain(c, quick):
    from scipy.stats import spearmanr
    names = list(GROUPS); n = len(names)
    masks = list(itertools.product([0, 1], repeat=n))
    feats = {}
    for m in masks:                                   # 1 = keep the group, 0 = neutralise it (replace by its mean)
        f = c.feat.clone()
        for g, name in enumerate(names):
            if not m[g]:
                f[:, GROUPS[name]] = f[:, GROUPS[name]].mean(0, keepdim=True)
        feats[m] = f

    def shapley(v):
        phi = np.zeros(n)
        for i in range(n):
            for m in masks:
                if m[i]:
                    continue
                s = sum(m); m2 = list(m); m2[i] = 1
                phi[i] += factorial(s) * factorial(n - s - 1) / factorial(n) * (v[tuple(m2)] - v[m])
        return phi

    X = np.array(masks, dtype=float)
    kern = np.sqrt(np.exp(-(np.sqrt((1 - X).sum(1)) ** 2) / (0.75 * np.sqrt(n)) ** 2))

    def lime(v):                                      # weighted ridge on on/off masks (LIME's tabular recipe)
        y = np.array([v[m] for m in masks])
        A = np.c_[np.ones(len(X)), X] * kern[:, None]
        coef = np.linalg.solve(A.T @ A + np.diag([0.0] + [1.0] * n), A.T @ (y * kern))
        return coef[1:]

    rng = np.random.default_rng(0)
    pool = [x for x in c.test_cases if len(x[1]) >= 3]
    pick = rng.choice(len(pool), min(30 if quick else 120, len(pool)), replace=False)
    all1, all0 = tuple([1] * n), tuple([0] * n)
    rho_se, rho_sl, top_se, top_sl, sign_ok, eff_err = [], [], [], [], [], []
    for k in pick:
        prefix = pool[k][1]
        vec = {m: c.svc._scores([prefix], feats[m])[0].numpy() for m in masks}
        full = vec[all1].copy(); full[prefix] = -1e9
        item = int(np.argmax(full))
        v = {m: float(vec[m][item]) for m in masks}
        phi, lim = shapley(v), lime(v)
        eff = np.array([v[all1] - v[tuple(0 if j == i else 1 for j in range(n))] for i in range(n)])  # built-in effects
        eff_err.append(abs(phi.sum() - (v[all1] - v[all0])))
        for a, b, store in ((phi, eff, rho_se), (phi, lim, rho_sl)):
            r = spearmanr(a, b).correlation
            if not np.isnan(r):
                store.append(r)
        top_se.append(np.argmax(np.abs(phi)) == np.argmax(np.abs(eff)))
        top_sl.append(np.argmax(np.abs(phi)) == np.argmax(np.abs(lim)))
        big = np.abs(phi) > 0.05
        if big.any():
            sign_ok.append(np.mean(np.sign(phi[big]) == np.sign(eff[big])))
    res = {"cases": len(pick),
           "shapley_efficiency_max_error": float(max(eff_err)),
           "spearman_shapley_vs_builtin": float(np.mean(rho_se)),
           "spearman_shapley_vs_lime": float(np.mean(rho_sl)),
           "top1_agree_shapley_builtin": float(np.mean(top_se)),
           "top1_agree_shapley_lime": float(np.mean(top_sl)),
           "sign_agree_shapley_builtin": float(np.mean(sign_ok))}
    log("EXPLAIN  " + json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in res.items()}))
    return res


# --------------------------------------------------------------------------
# 2. drift: damage, detection, incremental update
# --------------------------------------------------------------------------
def psi(ref, new, bins):
    r = np.histogram(ref, bins=bins)[0] / max(len(ref), 1) + 1e-4
    n = np.histogram(new, bins=bins)[0] / max(len(new), 1) + 1e-4
    return float(((n - r) * np.log(n / r)).sum())


def window_alarm(hits, ref_n, W=300, z_thr=3.0):
    """First index where the mean hit rate of the last W cases is z_thr sigmas below the reference rate."""
    p = hits[:ref_n].mean()
    sd = np.sqrt(p * (1 - p) / W)
    cs = np.concatenate([[0], np.cumsum(hits)])
    for i in range(ref_n + W, len(hits) + 1):
        if ((cs[i] - cs[i - W]) / W - p) / sd < -z_thr:
            return i - 1
    return None


def exp_drift(c, quick):
    ds = c.ds
    rng = np.random.default_rng(0)
    post = simulate_sessions(ds, regime=2, n_sessions=3, seed=123)
    control = simulate_sessions(ds, regime=0, n_sessions=2, seed=124)
    adapt_cases = [x for u in range(ds.n_users) for x in cases_from({u: post[u][0]})]
    eval_cases = [x for u in range(ds.n_users) for s in post[u][1:] for x in cases_from({u: s})]
    ctrl_cases = [x for u in range(ds.n_users) for s in control[u] for x in cases_from({u: s})]
    pre_cases = c.test_cases

    def scores_of(m, cases):
        return te.score_hybrid(m, cases, c.te_eu, c.te_ei, ds.n_users)

    def metr(m, cases):
        r = te.rank_metrics(scores_of(m, cases), np.array([x[2] for x in cases]))
        return {"HR@10": r["HR@10"], "NDCG@10": r["NDCG@10"]}

    res = {"frozen_pre": metr(c.model, pre_cases), "frozen_post": metr(c.model, eval_cases)}
    res["ndcg10_drop_pct"] = 100 * (1 - res["frozen_post"]["NDCG@10"] / res["frozen_pre"]["NDCG@10"])

    # --- label-free drift signal: how different are the items users now click?
    it = ds.meta["it"]
    ref_t = np.array([x[2] for x in pre_cases]); new_t = np.array([x[2] for x in eval_cases])
    res["psi"] = {"sentiment": psi(it["sentiment"][ref_t], it["sentiment"][new_t], np.linspace(-1, 1, 9)),
                  "abs_momentum": psi(np.abs(it["momentum"][ref_t]), np.abs(it["momentum"][new_t]), np.linspace(0, 1, 6)),
                  "market_cap_tier": psi(ds.item_tier[ref_t], ds.item_tier[new_t], [-.5, .5, 1.5, 2.5]),
                  "sector": psi(ds.item_sector[ref_t], ds.item_sector[new_t], np.arange(-.5, 8.5, 1))}

    # --- supervised monitor on the stream of hit@10 outcomes (clicks are the labels, they arrive immediately)
    def hits_of(cases):
        sc = scores_of(c.model, cases); tg = np.array([x[2] for x in cases])
        return ((sc >= sc[np.arange(len(tg)), tg][:, None]).sum(1) <= 10).astype(float)
    h_pre, h_post, h_ctrl = hits_of(pre_cases), hits_of(eval_cases), hits_of(ctrl_cases)
    rng.shuffle(h_pre); rng.shuffle(h_post); rng.shuffle(h_ctrl)
    ref_n = len(h_pre) // 2
    stream = np.concatenate([h_pre, h_post]); a = window_alarm(stream, ref_n)
    a_ctrl = window_alarm(np.concatenate([h_pre, h_ctrl]), ref_n)
    res["detection"] = {"reference_cases": int(ref_n), "shift_at_index": int(len(h_pre)),
                        "alarm_index": a, "false_alarm_before_shift": bool(a is not None and a < len(h_pre)),
                        "delay_cases_after_shift": None if a is None else int(a - len(h_pre)),
                        "control_no_drift_alarm": a_ctrl is not None}

    # --- incremental update
    idx = rng.permutation(len(adapt_cases)); cut = int(0.88 * len(idx))
    a_train = [adapt_cases[i] for i in idx[:cut]]; a_val = [adapt_cases[i] for i in idx[cut:]]
    ep = 1 if quick else 5

    def finetune(replay):
        m = copy.deepcopy(c.model)
        data = a_train + ([c.train_cases[i] for i in rng.choice(len(c.train_cases), int(0.3 * len(c.train_cases)), replace=False)]
                          if replay else [])
        m = te.train_session_model(m, ds, data, a_val, ep, 1e-3, 3, log, edges=(c.tr_eu, c.tr_ei), hybrid=True)
        return m.eval()

    variants = {"fine-tune on new data": finetune(False), "fine-tune + 30% replay": finetune(True)}
    scratch = PulseHybrid(ds.n_users, ds.n_items, ds.item_feat, use_gate=False)
    scratch = te.train_session_model(scratch, ds, a_train, a_val, 2 if quick else 15, 2e-3, 3, log,
                                     edges=(c.tr_eu, c.tr_ei), hybrid=True).eval()
    variants["retrain from scratch on new data only"] = scratch
    res["update"] = {"frozen (no update)": {"post": res["frozen_post"], "pre": res["frozen_pre"]}}
    for name, m in variants.items():
        res["update"][name] = {"post": metr(m, eval_cases), "pre": metr(m, pre_cases)}
    res["adaptation_cases"] = len(a_train)
    log("DRIFT    drop %.1f%% | PSI %s | detection %s" % (res["ndcg10_drop_pct"],
        {k: round(v, 2) for k, v in res["psi"].items()}, res["detection"]))
    for k, v in res["update"].items():
        log(f"         {k:40s} post NDCG@10 {v['post']['NDCG@10']:.4f}   pre NDCG@10 {v['pre']['NDCG@10']:.4f}")
    return res


# --------------------------------------------------------------------------
# 3. robustness: manipulated headlines, click injection, bounding defence
# --------------------------------------------------------------------------
def clamp_sm(f, cap):
    """Bound sentiment/momentum to [-cap, cap] (out-of-place, so gradients can flow through it)."""
    return torch.cat([f[:, :11], f[:, 11:13].clamp(-cap, cap)], 1)


def exp_adversarial(c, quick):
    rng = np.random.default_rng(1)
    # victim: the mid-cap whose news tone + momentum are currently LOWEST, so the attacker has the most room to push
    mids = [i for i, u in enumerate(UNIVERSE) if u[3] == 2]
    tone = {i: float(c.feat[i * STORIES_PER_TICKER:(i + 1) * STORIES_PER_TICKER, 11:13].mean()) for i in mids}
    target = UNIVERSE[min(tone, key=tone.get)][0]
    T = [SYMBOL_INDEX[target] * STORIES_PER_TICKER + k for k in range(STORIES_PER_TICKER)]
    enc, content, i_prop = c.model.enc, c.model.content, c.i_prop
    for p in list(enc.parameters()) + list(content.parameters()):
        p.requires_grad_(False)

    pool = [x[1] for x in c.test_cases if len(x[1]) >= 2 and not set(x[1].tolist()) & set(T)]
    order = rng.permutation(len(pool))
    n_eval, n_sur = (60, 40) if quick else (200, 100)
    evals = [pool[i] for i in order[:n_eval]]; sur = [pool[i] for i in order[n_eval:n_eval + n_sur]]

    def exposure(feat, prefixes):
        raw = serve_scores(enc, content, i_prop, feat, prefixes)
        raw_hit = dep_hit = 0
        ranks = []
        for p, row in zip(prefixes, raw):
            row = row.copy(); row[p] = -1e9
            raw_hit += bool(set(np.argsort(-row)[:10]) & set(T))
            ranks.append(1 + int((row > row[T].max()).sum()))          # rank of the target's best story among all items
            recs = c.svc.recommend([p], feat.numpy(), top_k=10, exclude=list(p), explain=False)
            dep_hit += bool({r["index"] for r in recs} & set(T))
        return raw_hit / len(prefixes), dep_hit / len(prefixes), float(np.mean(ranks))

    def pgd(eps, cap, steps=12):
        base = c.feat.clone(); delta = torch.zeros(len(T), 2)
        for _ in range(steps):
            delta.requires_grad_(True)
            f = base.clone(); f[T, 11:13] = (base[T, 11:13] + delta).clamp(-1, 1)
            f = clamp_sm(f, cap)
            raw = serve_scores_grad(enc, content, i_prop, f, sur)
            loss = -torch.logsumexp(raw[:, T], 1).mean()        # push up the strongest of the target's stories
            g, = torch.autograd.grad(loss, delta)
            delta = (delta.detach() - (eps / 4) * g.sign()).clamp(-eps, eps)
        f = c.feat.clone(); f[T, 11:13] = (c.feat[T, 11:13] + delta).clamp(-1, 1)
        return clamp_sm(f, cap)

    rows = []
    sub = c.test_cases[::3][:600 if quick else 1500]
    for cap in (1.0, 0.5, 0.25):
        fc = clamp_sm(c.feat, cap)
        clean = te.rank_metrics(serve_scores(enc, content, i_prop, fc, [x[1] for x in sub]), np.array([x[2] for x in sub]))["NDCG@10"]
        attacks = {"no attack": fc}
        naive = c.feat.clone(); naive[T, 11:13] = 1.0
        attacks["unbounded pump (set tone and momentum to +1)"] = clamp_sm(naive, cap)
        for eps in (0.5, 1.0, 2.0):                          # PGD may move each feature by at most eps from its current value
            attacks[f"PGD universal perturbation eps={eps}"] = pgd(eps, cap)
        for name, f in attacks.items():
            r, d, mr = exposure(f, evals)
            rows.append({"bound": cap, "attack": name, "target_in_top10_raw": r, "target_in_top10_deployed": d,
                         "target_mean_best_rank": mr, "clean_ndcg10": clean})
        for m in (1, 2, 4):                                  # click injection: fake reads of the target appended to the session
            inj = [np.concatenate([p, np.full(m, T[0])]).astype(np.int64) for p in evals]
            raw = serve_scores(enc, content, i_prop, fc, inj)
            rh = dh = 0
            rk = []
            for p, row in zip(inj, raw):
                row = row.copy(); row[p] = -1e9
                rh += bool(set(np.argsort(-row)[:10]) & set(T))
                rk.append(1 + int((row > row[T].max()).sum()))
                dh += bool({r_["index"] for r_ in c.svc.recommend([p], fc.numpy(), top_k=10, exclude=list(p), explain=False)} & set(T))
            rows.append({"bound": cap, "attack": f"click injection x{m}", "target_in_top10_raw": rh / len(inj),
                         "target_in_top10_deployed": dh / len(inj), "target_mean_best_rank": float(np.mean(rk)), "clean_ndcg10": clean})
    log(f"ADVERSARIAL  target={target}, {len(evals)} victim sessions")
    for r in rows:
        log(f"   bound {r['bound']:<4} {r['attack']:42s} raw top-10 {r['target_in_top10_raw']:.2f}   deployed top-10 {r['target_in_top10_deployed']:.2f}   mean best rank {r['target_mean_best_rank']:6.1f}   clean NDCG@10 {r['clean_ndcg10']:.4f}")
    return {"target": target, "victim_sessions": len(evals), "rows": rows}


# --------------------------------------------------------------------------
# 4. privacy: can the served list reveal a user's persona? noise as a defence
# --------------------------------------------------------------------------
def exp_privacy(c, quick):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    rng = np.random.default_rng(2)
    ds = c.ds; it = ds.meta["it"]
    by_user = {}
    for x in c.test_cases:
        if len(x[1]) >= 3:
            by_user.setdefault(x[0], []).append(x)
    users = sorted(by_user)[:300 if quick else 1000]
    picks = [by_user[u][rng.integers(len(by_user[u]))] for u in users]
    prefixes = [p[1] for p in picks]; targets = np.array([p[2] for p in picks]); persona = ds.user_persona[users]
    raw = serve_scores(c.model.enc, c.model.content, c.i_prop, c.feat, prefixes)
    for i, p in enumerate(prefixes):
        raw[i, p] = -1e9
    mask = raw > -1e8
    mu = np.array([r[m].mean() for r, m in zip(raw, mask)])[:, None]; sd = np.array([r[m].std() for r, m in zip(raw, mask)])[:, None]
    z = np.where(mask, (raw - mu) / sd, -1e9)

    def features(top):
        X = []
        for row in top:
            X.append(np.concatenate([np.bincount(ds.item_sector[row], minlength=8), np.bincount(ds.item_tier[row], minlength=3),
                                     [it["sentiment"][row].mean(), np.abs(it["momentum"][row]).mean(), it["momentum"][row].mean()]]))
        return np.array(X, dtype=float)

    rows = []
    chance = float(np.bincount(persona).max() / len(persona))
    for sigma in (0.0, 0.5, 1.0, 2.0, 4.0):
        noisy = np.where(mask, z + sigma * rng.gumbel(size=z.shape), -1e9)     # Gumbel-top-k = exponential-mechanism style sampling
        top = np.argsort(-noisy, axis=1)[:, :10]
        util = te.topk_metrics(top, targets)
        acc = cross_val_score(LogisticRegression(max_iter=2000), features(top), persona, cv=5).mean()
        rows.append({"noise_sigma": sigma, "persona_inference_accuracy": float(acc), "ndcg10": util["NDCG@10"], "hr10": util["HR@10"]})
    log(f"PRIVACY  persona-inference attack on top-10 lists, chance = {chance:.3f}")
    for r in rows:
        log(f"   sigma {r['noise_sigma']:<4} attacker accuracy {r['persona_inference_accuracy']:.3f}   NDCG@10 {r['ndcg10']:.4f}   HR@10 {r['hr10']:.4f}")
    return {"users": len(users), "chance": chance, "rows": rows,
            "note": "noise levels are nominal; a formal differential-privacy guarantee would need bounded score sensitivity"}


# --------------------------------------------------------------------------
# 5. compression: pruning, low-rank, int8
# --------------------------------------------------------------------------
def n_bytes(obj):
    b = io.BytesIO(); torch.save(obj, b); return b.tell()


def prune_(module, sparsity):
    for p in module.parameters():
        if p.dim() == 2:
            k = max(1, int(sparsity * p.numel()))
            thr = p.detach().abs().flatten().kthvalue(k).values
            p.data[p.detach().abs() <= thr] = 0.0
    return module


def lowrank(module, r):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and min(child.weight.shape) > r and child.out_features > 1:
            U, S, Vh = torch.linalg.svd(child.weight.data, full_matrices=False)
            a = nn.Linear(child.in_features, r, bias=False); a.weight.data = (S[:r, None] * Vh[:r]).contiguous()
            b = nn.Linear(r, child.out_features, bias=child.bias is not None); b.weight.data = U[:, :r].contiguous()
            if child.bias is not None:
                b.bias.data = child.bias.data.clone()
            setattr(module, name, nn.Sequential(a, b))
        else:
            lowrank(child, r)
    return module


def n_params(*mods, extra=0):
    return sum(p.numel() for m in mods for p in m.parameters()) + extra


def nonzero(*mods, extra=0):
    return sum(int((p != 0).sum()) for m in mods for p in m.parameters()) + extra


def exp_compress(c, quick):
    sub = c.test_cases[::3][:600 if quick else 1500]
    prefixes = [x[1] for x in sub]; targets = np.array([x[2] for x in sub])
    lat_set = prefixes[:60 if quick else 200]

    def measure(name, enc, content, i_prop, size, params, nnz=None):
        sc = serve_scores(enc, content, i_prop, c.feat, prefixes)
        nd = te.rank_metrics(sc, targets)["NDCG@10"]
        times = []
        for p in lat_set:
            t = time.perf_counter(); serve_scores(enc, content, i_prop, c.feat, [p]); times.append(time.perf_counter() - t)
        return {"variant": name, "ndcg10": nd, "size_kb": size / 1024, "parameters": params,
                "nonzero_parameters": nnz if nnz is not None else params, "p50_latency_ms": 1000 * float(np.median(times))}

    base_enc, base_con, ip = c.model.enc, c.model.content, c.i_prop
    rows = [measure("fp32 baseline", base_enc, base_con, ip, n_bytes([base_enc.state_dict(), base_con.state_dict(), ip]),
                    n_params(base_enc, base_con, extra=ip.numel()))]
    for s in (0.3, 0.5, 0.7, 0.9):
        e, k = prune_(copy.deepcopy(base_enc), s), prune_(copy.deepcopy(base_con), s)
        rows.append(measure(f"magnitude prune {int(s * 100)}%", e, k, ip, n_bytes([e.state_dict(), k.state_dict(), ip]),
                            n_params(e, k, extra=ip.numel()), nonzero(e, k, extra=ip.numel())))
    for r in (16, 8):
        e, k = lowrank(copy.deepcopy(base_enc), r), lowrank(copy.deepcopy(base_con), r)
        rows.append(measure(f"low-rank (SVD) rank {r}", e, k, ip, n_bytes([e.state_dict(), k.state_dict(), ip]), n_params(e, k, extra=ip.numel())))
    try:
        eq = torch.ao.quantization.quantize_dynamic(copy.deepcopy(base_enc), {nn.Linear}, dtype=torch.qint8)
        kq = torch.ao.quantization.quantize_dynamic(copy.deepcopy(base_con), {nn.Linear}, dtype=torch.qint8)
        rows.append(measure("int8 dynamic quantisation (layers)", eq, kq, ip, n_bytes([eq.state_dict(), kq.state_dict(), ip]),
                            n_params(base_enc, base_con, extra=ip.numel())))
    except Exception as e:
        rows.append({"variant": "int8 dynamic quantisation (layers)", "error": f"{type(e).__name__}: {e}"[:120]})
    scale = ip.abs().amax(1, keepdim=True) / 127.0
    q = torch.round(ip / scale).to(torch.int8); ip8 = q.float() * scale                 # per-row symmetric int8 table
    rows.append(measure("int8 item-embedding table", base_enc, base_con, ip8,
                        n_bytes([base_enc.state_dict(), base_con.state_dict(), q, scale]), n_params(base_enc, base_con, extra=ip.numel())))
    log("COMPRESS")
    for r in rows:
        if "error" in r:
            log(f"   {r['variant']:38s} not supported here: {r['error']}")
        else:
            log(f"   {r['variant']:38s} NDCG@10 {r['ndcg10']:.4f}  size {r['size_kb']:7.1f} KB  nonzero {r['nonzero_parameters']:>7d}/{r['parameters']:<7d}  p50 {r['p50_latency_ms']:.2f} ms")
    return {"cases": len(sub), "rows": rows,
            "note": "pruned models are stored dense here; the saving shows as nonzero parameters and needs a sparse format to realise"}


# --------------------------------------------------------------------------
# 6. behavioural segmentation into personas
# --------------------------------------------------------------------------
def exp_personas(c, quick):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    from sklearn.preprocessing import StandardScaler
    from scipy.optimize import linear_sum_assignment
    ds = c.ds; it = ds.meta["it"]
    users = [u for u in range(ds.n_users) if not ds.cold_users[u]]
    X = []
    for u in users:
        items = np.concatenate(c.train[u])
        X.append(np.concatenate([np.bincount(ds.item_sector[items], minlength=8) / len(items),
                                 np.bincount(ds.item_tier[items], minlength=3) / len(items),
                                 [it["sentiment"][items].mean(), np.abs(it["sentiment"][items]).mean(),
                                  np.abs(it["momentum"][items]).mean(), it["momentum"][items].mean()]]))
    X = StandardScaler().fit_transform(np.array(X)); truth = ds.user_persona[users]
    sil = {}
    for k in range(2, 7):
        sil[k] = float(silhouette_score(X, KMeans(k, n_init=10, random_state=0).fit_predict(X)))
    lab = KMeans(3, n_init=10, random_state=0).fit_predict(X)
    ari = float(adjusted_rand_score(truth, lab))
    conf = np.zeros((3, 3), int)
    for a, b in zip(lab, truth):
        conf[a, b] += 1
    r, cc = linear_sum_assignment(-conf)
    acc = float(conf[r, cc].sum() / len(truth))
    names = ["risk_averse", "day_trader", "long_term"]
    profiles = []
    for k in range(3):
        m = np.array(users)[lab == k]; items = np.concatenate([np.concatenate(c.train[u]) for u in m])
        profiles.append({"cluster": k, "users": int(len(m)), "best_matching_true_persona": names[cc[list(r).index(k)]],
                         "mega_cap_share": float((ds.item_tier[items] == 0).mean()), "mid_cap_share": float((ds.item_tier[items] == 2).mean()),
                         "mean_abs_momentum": float(np.abs(it["momentum"][items]).mean()), "mean_abs_sentiment": float(np.abs(it["sentiment"][items]).mean()),
                         "top_sector": SECTORS[int(np.bincount(ds.item_sector[items], minlength=8).argmax())]})
    log(f"PERSONAS silhouette by k {({k: round(v, 3) for k, v in sil.items()})} | k=3: ARI {ari:.3f}, matched accuracy {acc:.3f} (chance 0.333)")
    for p in profiles:
        log("   " + json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in p.items()}))
    return {"users": len(users), "silhouette_by_k": sil, "ari_k3": ari, "matched_accuracy_k3": acc, "profiles": profiles}


EXPERIMENTS = {"explain": exp_explain, "drift": exp_drift, "adversarial": exp_adversarial,
               "privacy": exp_privacy, "compress": exp_compress, "personas": exp_personas}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=list(EXPERIMENTS), default=list(EXPERIMENTS))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    ctx = build_ctx(args.quick)
    results = {}
    path = OUT / ("extras_quick.json" if args.quick else "extras.json")
    for name in args.only:
        log(f"=== {name} ===")
        results[name] = EXPERIMENTS[name](ctx, args.quick)
        json.dump(results, open(path, "w"), indent=2, default=float)
    log(f"done. results written to {path}")


if __name__ == "__main__":
    main()
