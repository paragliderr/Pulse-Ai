"""
Inference-side copy of the Pulse-Hybrid building blocks (session graph batching and
the SR-GNN session encoder). Identical to experiments/models.py so that the exported
checkpoint loads here. If you change the encoder in experiments/models.py, change it
here too and re-run experiments/export_model.py.
"""
import numpy as np
import torch
import torch.nn as nn


def session_batch(prefixes):
    B = len(prefixes)
    n_max = max(len(set(p)) for p in prefixes)
    L_max = max(len(p) for p in prefixes)
    nodes = np.zeros((B, n_max), dtype=np.int64)
    alias = np.zeros((B, L_max), dtype=np.int64)
    mask = np.zeros((B, L_max), dtype=np.float32)
    a_in = np.zeros((B, n_max, n_max), dtype=np.float32)
    a_out = np.zeros((B, n_max, n_max), dtype=np.float32)
    lens = np.zeros(B, dtype=np.int64)
    for b, seq in enumerate(prefixes):
        seq = np.asarray(seq, dtype=np.int64)
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
    t = torch.from_numpy
    return dict(nodes=t(nodes), alias=t(alias), mask=t(mask), a_in=t(a_in), a_out=t(a_out), lens=t(lens))


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
        h = node_emb
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
        seq_h = torch.gather(h, 1, alias)
        last = seq_h[torch.arange(seq_h.size(0)), batch["lens"] - 1]
        q = self.att_3(torch.sigmoid(self.att_1(last).unsqueeze(1) + self.att_2(seq_h)))
        a = (q * seq_h * batch["mask"].unsqueeze(-1)).sum(1)
        return self.out(torch.cat([a, last], -1))
