"""Prints the fairness, re-ranking and explanation numbers (averaged over seeds) in a short,
paste-friendly form.   Usage:  python show_extra_results.py [results/results.json]"""
import json, sys
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "results/results.json"
data = json.load(open(path))
runs = data["per_seed"]
avg = lambda vals: float(np.mean(vals))

print("== Exposure (top-10 share by market-cap tier; mean over seeds) ==")
print(f"{'source':28s} {'mega':>7s} {'large':>7s} {'mid':>7s} {'coverage@10':>12s}")
for src in runs[0]["fairness"]:
    f = [r["fairness"][src] for r in runs]
    cov = f"{avg([x['coverage@10'] for x in f]):12.3f}" if "coverage@10" in f[0] else f"{'-':>12s}"
    print(f"{src:28s} {avg([x['mega_share@10'] for x in f]):7.3f} {avg([x['large_share@10'] for x in f]):7.3f} "
          f"{avg([x['mid_share@10'] for x in f]):7.3f} {cov}")

print("\n== Re-ranking trade-off (Pulse-Hybrid, gated; mean over seeds) ==")
print(f"{'lambda':>7s} {'HR@10':>7s} {'NDCG@10':>8s} {'mega':>7s} {'mid':>7s} {'coverage':>9s}")
for i, row in enumerate(runs[0]["tradeoff"]):
    t = [r["tradeoff"][i] for r in runs]
    print(f"{row['lambda']:7.2f} {avg([x['HR@10'] for x in t]):7.4f} {avg([x['NDCG@10'] for x in t]):8.4f} "
          f"{avg([x['mega_share@10'] for x in t]):7.3f} {avg([x['mid_share@10'] for x in t]):7.3f} "
          f"{avg([x['coverage@10'] for x in t]):9.3f}")

print("\n== Explanation (seed 0) ==")
ex = runs[0]["explanations"]
print("effective long-term share of the user vector:", {k: round(v, 3) for k, v in ex["mean_long_term_share"].items()})
for who in ("warm_user", "cold_user"):
    e = ex[who]
    print(f"\n{who}: prefix length {e['prefix_len']}, history available: {e['history_available']}, "
          f"long-term share {e['long_term_share_of_user_vector']:.3f}")
    for r in e["top_recommendations"][:2]:
        d = {k: round(v, 2) for k, v in r["decomposition"].items()}
        g = {k: round(v, 2) for k, v in r["feature_group_effect"].items()}
        print(f"  item {r['item']} (ticker {r['ticker']}) score {r['score']:.2f}  actual next click: {r['is_actual_next_click']}")
        print(f"    decomposition: {d}")
        print(f"    feature-group effect: {g}")