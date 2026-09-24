"""
Builds the bipartite user-item interaction graph and its symmetrically
normalized adjacency matrix, as LightGCN expects. See Milestone 2 report,
Section 4 (Preprocessing Techniques — Graph construction).
"""

import numpy as np
import scipy.sparse as sp
import torch


def build_interaction_matrix(interactions: list[tuple[int, int]], n_users: int, n_items: int) -> sp.csr_matrix:
    """interactions: list of (user_idx, item_idx) pairs (positive interactions only)."""
    rows, cols = zip(*interactions) if interactions else ([], [])
    data = np.ones(len(rows))
    return sp.csr_matrix((data, (rows, cols)), shape=(n_users, n_items))


def k_core_filter(interaction_matrix: sp.csr_matrix, k: int = 5) -> sp.csr_matrix:
    """
    Iteratively drops users/items with fewer than k interactions, since
    very sparse users/items add noise without adding much signal to graph
    propagation. A few passes are needed since removing one side can push
    the other below the threshold too.
    """
    mat = interaction_matrix.copy()
    while True:
        user_counts = np.asarray(mat.sum(axis=1)).flatten()
        item_counts = np.asarray(mat.sum(axis=0)).flatten()
        keep_users = user_counts >= k
        keep_items = item_counts >= k
        if keep_users.all() and keep_items.all():
            break
        mat = mat[keep_users][:, keep_items]
    return mat


def build_normalized_adjacency(interaction_matrix: sp.csr_matrix) -> torch.sparse.Tensor:
    """
    Builds the full (n_users + n_items) x (n_users + n_items) adjacency
    matrix and applies symmetric normalization: D^-1/2 A D^-1/2, as
    required by LightGCN's propagation rule.
    """
    n_users, n_items = interaction_matrix.shape
    n = n_users + n_items

    adj = sp.lil_matrix((n, n))
    adj[:n_users, n_users:] = interaction_matrix
    adj[n_users:, :n_users] = interaction_matrix.T
    adj = adj.tocsr()

    degrees = np.asarray(adj.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(degrees, -0.5, where=degrees > 0)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    norm_adj = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
    norm_adj = norm_adj.tocoo()

    indices = torch.tensor(np.vstack([norm_adj.row, norm_adj.col]), dtype=torch.long)
    values = torch.tensor(norm_adj.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, size=norm_adj.shape)
