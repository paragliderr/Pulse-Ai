"""
LightGCN — long-term collaborative-filtering branch.

Reference: He et al., "LightGCN: Simplifying and Powering Graph Convolution
Network for Recommendation", SIGIR 2020.

Propagates user/item embeddings across the bipartite user-item interaction
graph (built in app/graph/build_graph.py) using simple linear propagation
(no feature transform, no nonlinearity), then combines the embeddings from
each layer to produce the final representation used for ranking.
"""

import torch
import torch.nn as nn


class LightGCN(nn.Module):
    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 64, n_layers: int = 3):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def forward(self, norm_adj: torch.sparse.Tensor):
        """
        norm_adj: symmetrically-normalized adjacency matrix of the full
        user-item graph (see app/data/preprocessing/graph.py for construction).

        Returns final user and item embeddings after layer propagation
        and combination (mean of all layers, per the LightGCN paper).
        """
        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        embeddings_per_layer = [all_embeddings]

        for _ in range(self.n_layers):
            all_embeddings = torch.sparse.mm(norm_adj, all_embeddings)
            embeddings_per_layer.append(all_embeddings)

        final_embeddings = torch.stack(embeddings_per_layer, dim=0).mean(dim=0)
        user_final, item_final = torch.split(final_embeddings, [self.n_users, self.n_items])
        return user_final, item_final

    # TODO: BPR loss training loop (see LightGCN paper, Section 3.3) — belongs
    # in a separate train_lightgcn.py script once the interaction graph is
    # populated from real/simulated data.
