"""
Synthetic data generator for Pulse AI experiments (Milestone 3).

Why synthetic: no public dataset of personalised financial-news interactions
exists, and real interaction logs are unavailable. This simulator builds a
universe of tickers/stories and a population of investor personas whose
click behaviour follows documented rules, so that every model is compared on
identical data and the results are fully reproducible without API keys.

Limitation (state it in the report): results validate the *method* under the
simulator's assumptions. They are not evidence of real-world user outcomes.
"""

from dataclasses import dataclass, field
import numpy as np

SECTORS = ["Technology", "Healthcare", "Financials", "Energy",
           "Consumer", "Industrials", "Utilities", "Materials"]
TIERS = ["mega", "large", "mid"]          # market-cap tiers (0, 1, 2)
PERSONAS = ["risk_averse", "day_trader", "long_term"]


@dataclass
class Dataset:
    n_users: int
    n_items: int
    item_ticker: np.ndarray      # (n_items,) ticker id of each story
    item_sector: np.ndarray      # (n_items,)
    item_tier: np.ndarray        # (n_items,) 0=mega 1=large 2=mid
    item_feat: np.ndarray        # (n_items, F) content features
    feat_groups: dict            # name -> column indices (used for explanations)
    user_persona: np.ndarray     # (n_users,)
    sessions: list = field(default_factory=list)   # per user: list of np.ndarray (chronological)
    cold_users: np.ndarray = None                  # bool mask (n_users,)


def _build_items(rng, n_tickers=40, stories_per_ticker=10):
    sector_of_ticker = np.repeat(np.arange(len(SECTORS)), n_tickers // len(SECTORS))
    rng.shuffle(sector_of_ticker)
    tier_of_ticker = np.array([0] * 10 + [1] * 18 + [2] * (n_tickers - 28))
    rng.shuffle(tier_of_ticker)

    ticker_mood = rng.normal(0, 0.45, n_tickers)       # market sentiment per ticker
    ticker_mom = rng.normal(0, 0.45, n_tickers)        # price momentum per ticker
    tier_pop = np.array([1.2, 0.5, 0.0])               # log-popularity bump per tier

    item_ticker = np.repeat(np.arange(n_tickers), stories_per_ticker)
    n_items = len(item_ticker)
    sentiment = np.clip(ticker_mood[item_ticker] + rng.normal(0, 0.35, n_items), -1, 1)
    momentum = np.clip(ticker_mom[item_ticker] + rng.normal(0, 0.30, n_items), -1, 1)

    item_sector = sector_of_ticker[item_ticker]
    item_tier = tier_of_ticker[item_ticker]
    item_logpop = tier_pop[item_tier] + rng.normal(0, 0.35, n_items)

    sector_oh = np.eye(len(SECTORS))[item_sector]
    tier_oh = np.eye(len(TIERS))[item_tier]
    feat = np.concatenate([sector_oh, tier_oh, sentiment[:, None], momentum[:, None]], axis=1)
    groups = {
        "sector": list(range(0, 8)),
        "market_cap": list(range(8, 11)),
        "sentiment": [11],
        "momentum": [12],
    }
    return dict(item_ticker=item_ticker, item_sector=item_sector, item_tier=item_tier,
                feat=feat.astype(np.float32), groups=groups, sentiment=sentiment,
                momentum=momentum, logpop=item_logpop, n_items=n_items,
                n_tickers=n_tickers)


def _persona_item_scores(persona, it, rng, fav_sectors, watch_tickers):
    """Long-term preference score of one user for every item."""
    s = np.zeros(it["n_items"])
    sec, tier = it["item_sector"], it["item_tier"]
    sent, mom = it["sentiment"], it["momentum"]
    if persona == 0:      # risk averse: defensive sectors, mega caps, calm names
        s += np.isin(sec, [6, 1, 2, 4]) * 1.0
        s += np.array([1.0, 0.4, -0.6])[tier]
        s += 0.3 * sent - 0.4 * np.abs(mom)
    elif persona == 1:    # day trader: volatile, high-momentum, strong sentiment
        s += np.isin(sec, [0, 3]) * 0.8
        s += np.array([0.0, 0.2, 0.6])[tier]
        s += 0.8 * np.abs(sent) + 0.9 * np.abs(mom)
    else:                 # long-term holder: own favourite sectors, quality names
        s += np.array([0.7, 0.4, -0.2])[tier]
        s += 0.1 * sent
    s += np.isin(sec, fav_sectors) * 0.9
    s += np.isin(it["item_ticker"], watch_tickers) * 1.5
    s += 0.5 * it["logpop"]
    return s


def generate(seed=0, n_users=1000, cold_frac=0.2, min_sessions=5, max_sessions=11):
    rng = np.random.default_rng(seed)
    it = _build_items(rng)
    n_items = it["n_items"]
    user_persona = rng.integers(0, len(PERSONAS), n_users)

    sessions_all = []
    for u in range(n_users):
        fav_sectors = rng.choice(len(SECTORS), 2, replace=False)
        watch_tickers = rng.choice(it["n_tickers"], 4, replace=False)
        base = _persona_item_scores(user_persona[u], it, rng, fav_sectors, watch_tickers)

        user_sessions = []
        n_sess = rng.integers(min_sessions, max_sessions + 1)
        for _ in range(n_sess):
            # each session has an intent that can differ from long-term taste
            r = rng.random()
            if r < 0.55:
                pool = watch_tickers if rng.random() < 0.6 else rng.choice(it["n_tickers"], 3)
                focus_t = int(rng.choice(pool)); focus_s = -1
            elif r < 0.85:
                focus_s = int(rng.choice(fav_sectors if rng.random() < 0.5 else np.arange(len(SECTORS))))
                focus_t = -1
            else:
                focus_t, focus_s = -1, -1

            length = int(rng.integers(4, 10))
            seq = []
            for _step in range(length):
                sc = base.copy()
                if focus_t >= 0:
                    sc += 2.2 * (it["item_ticker"] == focus_t)
                if focus_s >= 0:
                    sc += 1.6 * (it["item_sector"] == focus_s)
                if seq:                                    # story chains within a ticker
                    sc += 0.9 * (it["item_ticker"] == it["item_ticker"][seq[-1]])
                    sc[seq] -= 1.5                         # rarely re-click the same story
                p = np.exp(sc - sc.max()); p /= p.sum()
                seq.append(int(rng.choice(n_items, p=p)))
            user_sessions.append(np.array(seq, dtype=np.int64))
        sessions_all.append(user_sessions)

    cold = np.zeros(n_users, dtype=bool)
    cold[rng.choice(n_users, int(cold_frac * n_users), replace=False)] = True

    return Dataset(
        n_users=n_users, n_items=n_items,
        item_ticker=it["item_ticker"], item_sector=it["item_sector"],
        item_tier=it["item_tier"], item_feat=it["feat"], feat_groups=it["groups"],
        user_persona=user_persona, sessions=sessions_all, cold_users=cold,
    )


def split(ds: Dataset):
    """
    Chronological split per user:
      train = all sessions except the last two, val = second-to-last, test = last.
    Cold-start users have their train/val history removed entirely, so the only
    thing known about them at test time is the first part of their final session.
    """
    train, val, test = {}, {}, {}
    for u, sess in enumerate(ds.sessions):
        test[u] = sess[-1]
        if ds.cold_users[u]:
            train[u], val[u] = [], None
        else:
            val[u] = sess[-2]
            train[u] = sess[:-2]
    return train, val, test


def cases_from(session_by_user, users=None):
    """(user, prefix, target) for every position of every session given."""
    out = []
    for u, s in session_by_user.items():
        if s is None or (users is not None and u not in users):
            continue
        for k in range(1, len(s)):
            out.append((u, s[:k], int(s[k])))
    return out


def edges_from(train, val=None):
    """Unique (user, item) pairs from train (and optionally val) sessions."""
    pairs = set()
    for u, sess in train.items():
        for s in sess:
            for i in s:
                pairs.add((u, int(i)))
    if val is not None:
        for u, s in val.items():
            if s is not None:
                for i in s:
                    pairs.add((u, int(i)))
    arr = np.array(sorted(pairs), dtype=np.int64) if pairs else np.zeros((0, 2), dtype=np.int64)
    return arr[:, 0], arr[:, 1]