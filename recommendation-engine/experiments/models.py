"""
Models for the Pulse AI Milestone 3 experiments.

Baselines (state of the art, re-implemented from the papers):
  - BPRMF      : matrix factorisation with BPR loss
  - LightGCN   : He et al., SIGIR 2020  (long-term collaborative signal)
  - SRGNN      : Wu et al., AAAI 2019   (short-term session signal)

Proposed:
  - PulseHybrid: LightGCN user/item embeddings + SR-GNN session encoder,
                 content (sentiment/momentum/sector/market-cap) item features,
                 and a history-aware gate that decides, per user and per
                 request, how much to trust long-term vs. in-session intent.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# LightGCN propagation (no feature transform, no non-linearity)
# --------------------------------------------------------------------------
def propagate(user0, item0, eu, ei, n_layers):
    """
    Symmetric-normalised linear propagation on the bipartite user-item graph,
    followed by the mean over layers 0..L (as in the LightGCN paper).
    eu, ei: LongTensors with one entry per edge.
    """
    n_u, n_i = user0.shape[0], item0.shape[0]
    if eu.numel() == 0:
        return user0, item0
    deg_u = torch.bincount(eu, minlength=n_u).float()
    deg_i = torch.bincount(ei, minlength=n_i).float()
    w = (deg_u[eu].clamp(min=1) * deg_i[ei].clamp(min=1)).rsqrt().unsqueeze(1)

    u, i = user0, item0
    us, is_ = [user0], [item0]
    for _ in range(n_layers):
        new_u = torch.zeros_like(user0).index_add_(0, eu, i[ei] * w)
        new_i = torch.zeros_like(item0).index_add_(0, ei, u[eu] * w)
        u, i = new_u, new_i
        us.append(u)
        is_.append(i)
    return torch.stack(us).mean(0), torch.stack(is_).mean(0)


# --------------------------------------------------------------------------
# Session batching (SR-GNN style session graphs)
# --------------------------------------------------------------------------
def session_batch(prefixes):
    """
    Turn a list of item-id sequences into padded tensors:
      nodes (B,n)  unique items per session      a_in/a_out (B,n,n)
      alias (B,L)  position -> node index         mask (B,L)    lens (B,)
    """
    B = len(prefixes)
    n_max = max(len(set(p.tolist())) for p in prefixes)
    L_max = max(len(p) for p in prefixes)
    nodes = np.zeros((B, n_max), dtype=np.int64)
    alias = np.zeros((B, L_max), dtype=np.int64)
    mask = np.zeros((B, L_max), dtype=np.float32)
    a_in = np.zeros((B, n_max, n_max), dtype=np.float32)
    a_out = np.zeros((B, n_max, n_max), dtype=np.float32)
    lens = np.zeros(B, dtype=np.int64)
    for b, seq in enumerate(prefixes):
        uniq, inv = np.unique(seq, return_inverse=True)
        nodes[b, :len(uniq)] = uniq
        alias[b, :len(seq)] = inv
        mask[b, :len(seq)] = 1.0
        lens[b] = len(seq)
        A = np.zeros((len(uniq), len(uniq)), dtype=np.float32)
        for x, y in zip(inv[:-1], inv[1:]):
            A[x, y] += 1.0
        out_deg = A.sum(1, keepdims=True); out_deg[out_deg == 0] = 1
        in_deg = A.sum(0, keepdims=True); in_deg[in_deg == 0] = 1
        a_out[b, :len(uniq), :len(uniq)] = A / out_deg
        a_in[b, :len(uniq), :len(uniq)] = (A / in_deg).T
    t = lambda x: torch.from_numpy(x)
    return dict(nodes=t(nodes), alias=t(alias), mask=t(mask),
                a_in=t(a_in), a_out=t(a_out), lens=t(lens))


class SessionEncoder(nn.Module):
    """Gated graph neural network + soft attention (Wu et al., 2019)."""

    def __init__(self, d, steps=1):
        super().__init__()
        self.d, self.steps = d, steps
        self.lin_in = nn.Linear(d, d)
        self.lin_out = nn.Linear(d, d)
        self.w_ih = nn.Linear(2 * d, 3 * d)
        self.w_hh = nn.Linear(d, 3 * d)
        self.att_1 = nn.Linear(d, d)
        self.att_2 = nn.Linear(d, d, bias=False)
        self.att_3 = nn.Linear(d, 1, bias=False)
        self.out = nn.Linear(2 * d, d, bias=False)

    def forward(self, node_emb, batch):
        h = node_emb                                            # (B,n,d)
        for _ in range(self.steps):
            m_in = torch.bmm(batch["a_in"], self.lin_in(h))
            m_out = torch.bmm(batch["a_out"], self.lin_out(h))
            gi = self.w_ih(torch.cat([m_in, m_out], -1))
            gh = self.w_hh(h)
            i_r, i_z, i_n = gi.chunk(3, -1)
            h_r, h_z, h_n = gh.chunk(3, -1)
            r = torch.sigmoid(i_r + h_r)
            z = torch.sigmoid(i_z + h_z)
            n = torch.tanh(i_n + r * h_n)
            h = n + z * (h - n)

        alias = batch["alias"].unsqueeze(-1).expand(-1, -1, self.d)
        seq_h = torch.gather(h, 1, alias)                       # (B,L,d)
        last = seq_h[torch.arange(seq_h.size(0)), batch["lens"] - 1]
        q = self.att_3(torch.sigmoid(self.att_1(last).unsqueeze(1) + self.att_2(seq_h)))
        a = (q * seq_h * batch["mask"].unsqueeze(-1)).sum(1)    # global preference
        return self.out(torch.cat([a, last], -1))               # global + current


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------
class BPRMF(nn.Module):
    def __init__(self, n_users, n_items, d=64):
        super().__init__()
        self.user = nn.Embedding(n_users, d)
        self.item = nn.Embedding(n_items, d)
        nn.init.normal_(self.user.weight, std=0.1)
        nn.init.normal_(self.item.weight, std=0.1)

    def all_embeddings(self, eu=None, ei=None):
        return self.user.weight, self.item.weight


class LightGCN(BPRMF):
    def __init__(self, n_users, n_items, d=64, n_layers=2):
        super().__init__(n_users, n_items, d)
        self.n_layers = n_layers

    def all_embeddings(self, eu, ei):
        return propagate(self.user.weight, self.item.weight, eu, ei, self.n_layers)


class SRGNN(nn.Module):
    def __init__(self, n_items, d=64, steps=1):
        super().__init__()
        self.item = nn.Embedding(n_items, d)
        nn.init.normal_(self.item.weight, std=0.1)
        self.enc = SessionEncoder(d, steps)

    def forward(self, batch):
        s = self.enc(self.item(batch["nodes"]), batch)
        return s @ self.item.weight.T


# --------------------------------------------------------------------------
# Proposed model
# --------------------------------------------------------------------------
class PulseHybrid(nn.Module):
    """
    score(u, session, j) = z . item_j
      item_j = LightGCN_item_j + MLP(content_features_j)
      z      = g * LightGCN_user_u  +  (1 - g) * SRGNN(session; item table)
      g      = sigmoid(W [user ; session ; history_confidence])

    Flags allow clean ablations (each one removes exactly one idea).
    """

    def __init__(self, n_users, n_items, item_feat, d=64, n_layers=2, steps=1,
                 use_graph=True, use_session=True, use_content=True, use_gate=True):
        super().__init__()
        self.d, self.n_layers = d, n_layers
        self.use_graph, self.use_session = use_graph, use_session
        self.use_content, self.use_gate = use_content, use_gate

        self.user = nn.Embedding(n_users, d)
        self.item = nn.Embedding(n_items, d)
        nn.init.normal_(self.user.weight, std=0.1)
        nn.init.normal_(self.item.weight, std=0.1)
        self.register_buffer("feat", torch.as_tensor(item_feat))
        self.content = nn.Sequential(nn.Linear(self.feat.shape[1], d), nn.Tanh(), nn.Linear(d, d))
        self.enc = SessionEncoder(d, steps)
        self.gate = nn.Linear(2 * d + 1, d)

    def item_table(self, eu, ei):
        if self.use_graph:
            u_tab, i_prop = propagate(self.user.weight, self.item.weight, eu, ei, self.n_layers)
        else:
            u_tab, i_prop = None, self.item.weight
        content = self.content(self.feat) if self.use_content else torch.zeros_like(i_prop)
        return u_tab, i_prop, content

    def forward(self, users, batch, eu, ei, hist_mask, conf, parts=False):
        u_tab, i_prop, content = self.item_table(eu, ei)
        item_final = i_prop + content

        s = self.enc(item_final[batch["nodes"]], batch) if self.use_session else None
        u = (u_tab[users] * hist_mask.unsqueeze(1)) if self.use_graph else None

        if u is not None and s is not None:
            if self.use_gate:
                g = torch.sigmoid(self.gate(torch.cat([u, s, conf.unsqueeze(1)], -1)))
            else:
                g = torch.full_like(u, 0.5)
            z = g * u + (1 - g) * s
        elif u is not None:
            g, z = torch.ones_like(u), u
        else:
            g, z = torch.zeros_like(s), s

        scores = z @ item_final.T
        if not parts:
            return scores
        gu = g * u if u is not None else torch.zeros_like(z)
        gs = (1 - g) * s if s is not None else torch.zeros_like(z)
        return scores, dict(g=g, gu=gu, gs=gs, i_prop=i_prop, content=content)