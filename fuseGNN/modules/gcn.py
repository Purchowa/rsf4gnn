import torch
import torch.nn.functional as F
from fuseGNN.convs import geoGCNConv, refGCNConv, garGCNConv, gasGCNConv
from fuseGNN.utils import LrSchedular


modules = {
    'geo': geoGCNConv,
    'ref': refGCNConv,
    'gar': garGCNConv,
    'gas': gasGCNConv
}


class GCN(torch.nn.Module):
    """
    The graph convolutional operator from the `"Semi-supervised Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>
    A two-layer GCN model.
    The model is trained for 200 epochs with learning rate 0.01 and early stopped with a window size of 10
    (the validation loss doesn't decrease for 10 consecutive epochs)
    """
    def __init__(self, num_features, hidden, num_classes, cached=True, drop_rate=0.5, mode='geo', flow='source_to_target'):
        """
        :param num_features: the length of input features
        :param hidden: the length of hidden layer
        :param num_classes: the number of classes
        :param cached: If True, the layer will cache the computation on first execution, and will use the
        cached version for further executions. So it should be only true in transductive learning scenarios
        """
        super(GCN, self).__init__()
        self.GCNConv = modules[mode]
        self.mode = mode
        self.conv1 = self.GCNConv(in_channels=num_features, out_channels=hidden, cached=cached, flow=flow)
        self.conv2 = self.GCNConv(in_channels=hidden, out_channels=num_classes, cached=cached, flow=flow)
        
        self.reg_params = self.conv1.parameters()
        self.non_reg_params = self.conv2.parameters()
        self.drop_rate = drop_rate

    @staticmethod
    def _unpack_adj(adj):
        """Extract edge_index and bipartite size from sampled adjacency.

        Args:
            adj: PyG adjacency object (with ``edge_index`` and ``size``) or
                tuple representation ``(edge_index, e_id, size)``.

        Returns:
            tuple: ``(edge_index, size)`` used by sampled forward path.
        """
        if hasattr(adj, 'edge_index') and hasattr(adj, 'size'):
            return adj.edge_index, adj.size
        return adj[0], adj[2]
    
    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        if x.dtype != torch.float32:
            x = x.float()
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(input=x, p=self.drop_rate, training=self.training)
        if self.mode == 'gar':
            x = self.conv2(x=x, edge_weight=self.conv1.cached_edge_weight_f, self_edge_weight=self.conv1.cached_self_edge_weight,
                           tar_ptr=self.conv1.cached_tar_ptr, src_index=self.conv1.cached_src_index,
                           src_ptr=self.conv1.cached_src_ptr, tar_index=self.conv1.cached_tar_index,
                           edge_weight_b=self.conv1.cached_edge_weight_b)
        elif self.mode == 'gas':
            x = self.conv2(x=x, edge_weight=self.conv1.cached_edge_weight, src_index=self.conv1.cached_src_index,
                           tar_index=self.conv1.cached_tar_index, self_edge_weight=self.conv1.cached_self_edge_weight)
        else:
            x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)

    def forward_sampled(
        self,
        x,
        adjs,
        gar_sampling_variant='per_batch_csr',
        sampling_debug_enabled=False,
        sampling_debug_fail_fast=False,
        sampling_debug_split='unknown',
        sampling_debug_batch=-1,
    ):
        """Run two-layer GCN forward on a sampled mini-batch subgraph.

        This path is used by NeighborSampler-based training and evaluation.
        It avoids full-graph assumptions and uses per-batch structures.

        Args:
            x (torch.Tensor): Batch feature matrix indexed by sampled node ids.
            adjs (list): Two adjacency objects (one per GCN layer) from
                NeighborSampler.
            gar_sampling_variant (str): GAR mode selection. Accepted values are
                ``per_batch_csr`` and ``hybrid_full_cache``.
            sampling_debug_enabled (bool): Enables sampled debug traces.
            sampling_debug_fail_fast (bool): Enables fail-fast sampled invariants.
            sampling_debug_split (str): Split label for debug logs.
            sampling_debug_batch (int): Batch index for debug logs.

        Returns:
            torch.Tensor: Log-probabilities for sampled nodes.
        """
        if x.dtype != torch.float32:
            x = x.float()
        # NeighborSampler yields one adjacency per layer.
        adj_l1, adj_l2 = adjs[0], adjs[1]
        edge_index_l1, _ = self._unpack_adj(adj_l1)
        edge_index_l2, _ = self._unpack_adj(adj_l2)

        # Keep sampled edge indices on the same device as features so GAR/GAS
        # format conversion runs on GPU instead of CPU.
        edge_index_l1 = edge_index_l1.to(x.device)
        edge_index_l2 = edge_index_l2.to(x.device)

        if sampling_debug_enabled:
            e1_count = int(edge_index_l1.size(1))
            e2_count = int(edge_index_l2.size(1))
            e1_max = int(edge_index_l1.max().item()) if e1_count > 0 else -1
            e2_max = int(edge_index_l2.max().item()) if e2_count > 0 else -1
            print(
                '[SamplingDebug][GCN] split={} batch={} x_rows={} '
                'l1(dtype={},device={},contig={},edges={},max={}) '
                'l2(dtype={},device={},contig={},edges={},max={})'.format(
                    sampling_debug_split,
                    sampling_debug_batch,
                    int(x.size(0)),
                    str(edge_index_l1.dtype),
                    str(edge_index_l1.device),
                    edge_index_l1.is_contiguous(),
                    e1_count,
                    e1_max,
                    str(edge_index_l2.dtype),
                    str(edge_index_l2.device),
                    edge_index_l2.is_contiguous(),
                    e2_count,
                    e2_max,
                )
            )

        if sampling_debug_fail_fast:
            if edge_index_l1.numel() > 0 and int(edge_index_l1.max().item()) >= int(x.size(0)):
                raise RuntimeError(
                    'Layer1 sampled edge index out of bounds: split={} batch={} max_index={} x_rows={}'.format(
                        sampling_debug_split,
                        sampling_debug_batch,
                        int(edge_index_l1.max().item()),
                        int(x.size(0)),
                    )
                )
            if edge_index_l2.numel() > 0 and int(edge_index_l2.max().item()) >= int(x.size(0)):
                raise RuntimeError(
                    'Layer2 sampled edge index out of bounds: split={} batch={} max_index={} x_rows={}'.format(
                        sampling_debug_split,
                        sampling_debug_batch,
                        int(edge_index_l2.max().item()),
                        int(x.size(0)),
                    )
                )

        if self.mode in ['gar', 'gas']:
            x = F.relu(
                self.conv1(
                    x,
                    edge_index_l1,
                    None,
                    sampling_debug_enabled=sampling_debug_enabled,
                    sampling_debug_fail_fast=sampling_debug_fail_fast,
                    sampling_debug_split=sampling_debug_split,
                    sampling_debug_batch=sampling_debug_batch,
                    sampling_debug_layer=1,
                )
            )
        else:
            x = F.relu(self.conv1(x, edge_index_l1, None))
        x = F.dropout(input=x, p=self.drop_rate, training=self.training)

        if self.mode == 'gar':
            if gar_sampling_variant == 'hybrid_full_cache':
                # For sampled forward, fallback to per-batch COO->CSR/CSC conversion.
                x = self.conv2(
                    x=x,
                    edge_index=edge_index_l2,
                    edge_weight=None,
                    sampling_debug_enabled=sampling_debug_enabled,
                    sampling_debug_fail_fast=sampling_debug_fail_fast,
                    sampling_debug_split=sampling_debug_split,
                    sampling_debug_batch=sampling_debug_batch,
                    sampling_debug_layer=2,
                )
            else:
                x = self.conv2(
                    x=x,
                    edge_index=edge_index_l2,
                    edge_weight=None,
                    sampling_debug_enabled=sampling_debug_enabled,
                    sampling_debug_fail_fast=sampling_debug_fail_fast,
                    sampling_debug_split=sampling_debug_split,
                    sampling_debug_batch=sampling_debug_batch,
                    sampling_debug_layer=2,
                )
        elif self.mode == 'gas':
            x = self.conv2(
                x=x,
                edge_index=edge_index_l2,
                edge_weight=None,
                sampling_debug_enabled=sampling_debug_enabled,
                sampling_debug_fail_fast=sampling_debug_fail_fast,
                sampling_debug_split=sampling_debug_split,
                sampling_debug_batch=sampling_debug_batch,
                sampling_debug_layer=2,
            )
        else:
            x = self.conv2(x, edge_index_l2, None)
        return F.log_softmax(x, dim=1)


"""
    Training configurations from 
    "Semi-supervised Classification with Graph Convolutional Networks"
    <https://arxiv.org/pdf/1609.02907.pdf> The lines below as cited from the paper:
    > we train a two-layer GCN as ...
    > For the citation network datasets, we optimize hyperparameters on Cora only 
      and use the same set of parameters for Citeseer and Pubmed.
    > We train all models for a maximum of 200 epochs (training iterations) 
      using Adam with a learning rate of 0.01
    > We used the following sets of hyperparameters for Citeseer, Cora and Pubmed: 
      0.5 (dropout rate), 5 · 10−4 (L2 regularization) and 16 (number of hid- den units); 
      and for NELL: 0.1 (dropout rate), 1 · 10−5 (L2 regularization) and 64 (number of hidden units).
"""
gcn_config = {
    'CiteSeer': {
        'drop_rate': 0.5,
        'weight_decay': 4e-4,
        'hidden': 16,
        'lr': 0.01,
        'lr_schedular': LrSchedular(init_lr=0.01, mode='constant'),
        'fold': 1,
    },
    'Cora': {
        'drop_rate': 0.5,
        'weight_decay': 4e-4,
        'hidden': 16,
        'lr': 0.01,
        'lr_schedular': LrSchedular(init_lr=0.01, mode='constant'),
        'fold': 1,
    },
    'PubMed': {
        'drop_rate': 0.5,
        'weight_decay': 4e-4,
        'hidden': 16,
        'lr': 0.01,
        'lr_schedular': LrSchedular(init_lr=0.01, mode='constant'),
        'fold': 1,
    },
    'Nell': {
        'drop_rate': 0.1,
        'weight_decay': 1e-5,
        'hidden': 64,
        'lr': 0.01,
        'lr_schedular': LrSchedular(init_lr=0.01, mode='constant'),
        'fold': 1,
    },
    'Reddit': {
        'drop_rate': 0.5,
        'weight_decay': 1e-5,
        'hidden': 128,
        'lr': 0.01,
        'lr_schedular': LrSchedular(init_lr=0.01, mode='constant'),
        'fold': 1,
    },
    'ogbn-arxiv': {
        'drop_rate': 0.5,
        'weight_decay': 0.0,
        'hidden': 256,
        'lr': 0.01,
        'lr_schedular': LrSchedular(init_lr=0.01, mode='constant'),
        'fold': 1,
    },
    'ogbn-products': {
        'drop_rate': 0.5,
        'weight_decay': 0.0,
        'hidden': 256,
        'lr': 0.01,
        'lr_schedular': LrSchedular(init_lr=0.01, mode='constant'),
        'fold': 1,
    },
    'ogbn-papers100M': {
        'drop_rate': 0.5,
        'weight_decay': 0.0,
        'hidden': 256,
        'lr': 0.01,
        'lr_schedular': LrSchedular(init_lr=0.01, mode='constant'),
        'fold': 1,
    }
}
