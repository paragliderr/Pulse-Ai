"""
SR-GNN — short-term, session-based branch.

Reference: Wu et al., "Session-based Recommendation with Graph Neural
Networks", AAAI 2019.

Each session's click sequence is converted into a directed graph of
item-to-item transitions (see app/data/preprocessing/session.py). A gated
graph neural network learns item embeddings from this graph, and an
attention mechanism combines the session's global preference with its most
recent interest into a single session embedding used to score candidate
next items — this is the branch that handles cold-start users with no
long-term interaction history.
"""

import torch
import torch.nn as nn


class GatedSessionGraphConv(nn.Module):
    """One step of gated propagation over a session graph."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.gru_cell = nn.GRUCell(hidden_size, hidden_size)
        self.linear_in = nn.Linear(hidden_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden, adj_in, adj_out):
        a_in = torch.matmul(adj_in, self.linear_in(hidden))
        a_out = torch.matmul(adj_out, self.linear_out(hidden))
        inputs = a_in + a_out
        return self.gru_cell(inputs, hidden)


class SRGNN(nn.Module):
    def __init__(self, n_items: int, hidden_size: int = 100, n_steps: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_steps = n_steps

        self.item_embedding = nn.Embedding(n_items, hidden_size)
        self.gnn = GatedSessionGraphConv(hidden_size)

        # Attention over session items to combine global + local (last-click) intent
        self.attn_global = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_local = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, session_items: torch.Tensor, adj_in: torch.Tensor, adj_out: torch.Tensor):
        """
        session_items: item indices present in the session graph (n_nodes,)
        adj_in / adj_out: normalized in/out adjacency for this session graph
        Returns a single session embedding (hidden_size,).
        """
        hidden = self.item_embedding(session_items)
        for _ in range(self.n_steps):
            hidden = self.gnn(hidden, adj_in, adj_out)

        last_item_hidden = hidden[-1]  # most recent click in the session
        global_ctx = self.attn_global(hidden)
        local_ctx = self.attn_local(last_item_hidden).unsqueeze(0)
        alpha = torch.softmax(self.attn_score(torch.tanh(global_ctx + local_ctx)), dim=0)
        session_embedding = (alpha * hidden).sum(dim=0)
        return session_embedding

    # TODO: cross-entropy training loop over next-item prediction (see SR-GNN
    # paper, Section 3.3) — belongs in a separate train_srgnn.py script.
