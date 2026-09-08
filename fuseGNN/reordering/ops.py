"""
Graph reordering utilities for optimizing memory access patterns.
"""

import numpy as np
import torch

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee


def compute_degree_permutation(edge_index, num_nodes):
    src_index = edge_index[0]
    tar_index = edge_index[1]
    degree = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
    one_src = torch.ones(src_index.size(0), dtype=torch.long, device=edge_index.device)
    one_tar = torch.ones(tar_index.size(0), dtype=torch.long, device=edge_index.device)
    degree.index_add_(0, src_index, one_src)
    degree.index_add_(0, tar_index, one_tar)

    _, perm = torch.sort(degree, descending=True)
    perm_inv = torch.empty_like(perm)
    perm_inv[perm] = torch.arange(num_nodes, dtype=torch.long, device=edge_index.device)
    return perm, perm_inv


def compute_rcm_permutation(edge_index, num_nodes):
    """Compute a Reverse Cuthill-McKee node permutation.

    Args:
        edge_index (torch.Tensor): Edge index tensor with shape [2, num_edges].
        num_nodes (int): Number of nodes in the graph.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The permutation and its inverse.
    """
    
    device = edge_index.device
    edge_index_cpu = edge_index.detach().cpu()
    row = edge_index_cpu[0].numpy()
    col = edge_index_cpu[1].numpy()

    adjacency = csr_matrix(
        (np.ones(row.shape[0], dtype=np.int8), (row, col)),
        shape=(num_nodes, num_nodes),
    )
    adjacency = adjacency.maximum(adjacency.transpose())

    perm_np = reverse_cuthill_mckee(adjacency, symmetric_mode=True).copy()
    perm = torch.as_tensor(perm_np, dtype=torch.long, device=device)
    perm_inv = torch.empty_like(perm)
    perm_inv[perm] = torch.arange(num_nodes, dtype=torch.long, device=device)
    return perm, perm_inv


def reorder_node_tensors(data, perm):
    if data.x is not None:
        data.x = data.x[perm]
    if data.y is not None:
        data.y = data.y[perm]
    if data.train_mask is not None:
        data.train_mask = data.train_mask[perm]
    if data.val_mask is not None:
        data.val_mask = data.val_mask[perm]
    if data.test_mask is not None:
        data.test_mask = data.test_mask[perm]
    if hasattr(data, 'deg') and data.deg is not None:
        data.deg = data.deg[perm]


def reorder_edges(edge_index, edge_attr, perm_inv, flow):
    src_new = perm_inv[edge_index[0]]
    tar_new = perm_inv[edge_index[1]]
    edge_index_new = torch.stack([src_new, tar_new], dim=0)

    if flow == 'target_to_source':
        primary = edge_index_new[0]
        secondary = edge_index_new[1]
    else:
        primary = edge_index_new[1]
        secondary = edge_index_new[0]

    num_nodes = int(perm_inv.numel())
    sort_key = primary * (num_nodes + 1) + secondary
    sort_idx = torch.argsort(sort_key)

    edge_index_sorted = edge_index_new[:, sort_idx]
    if edge_attr is not None and edge_attr.size(0) == edge_index.size(1):
        edge_attr_sorted = edge_attr[sort_idx]
    else:
        edge_attr_sorted = edge_attr
    return edge_index_sorted, edge_attr_sorted