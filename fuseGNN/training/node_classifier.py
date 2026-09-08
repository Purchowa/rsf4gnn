"""Mini-batch and full-batch node classification helpers.

This module keeps training script orchestration small by moving epoch logic,
sampling utilities, and split-level metric aggregation into one reusable place.
"""

import time

import torch
import torch.nn.functional as F
from torch_geometric.data import NeighborSampler

from fuseGNN.utils import compute_acc_micro_macro_f1, safe_div


def parse_sizes(sizes_text):
    """Parse sampler neighborhood sizes from CLI text.

    Args:
        sizes_text (str): Comma-separated positive integers, for example "25,10".

    Returns:
        list[int]: Neighborhood sizes for each hop.

    Raises:
        ValueError: If input is empty or contains non-positive values.
    """
    values = [item.strip() for item in sizes_text.split(',') if item.strip()]
    if not values:
        raise ValueError('Sampling sizes must be a non-empty comma-separated list.')
    parsed = [int(item) for item in values]
    if any(v <= 0 for v in parsed):
        raise ValueError('Sampling sizes must be positive integers.')
    return parsed


def compute_scores_from_pred_labels(pred, true, num_classes):
    """Compute accuracy, micro F1, and macro F1 from class indices.

    Args:
        pred (torch.Tensor): Predicted class indices of shape [N].
        true (torch.Tensor): Ground-truth class indices of shape [N].
        num_classes (int): Number of classes in the task.

    Returns:
        tuple[float, float, float]: Accuracy, micro-F1, macro-F1.
    """
    total = int(true.numel())
    if total == 0:
        return 0.0, 0.0, 0.0

    correct = pred.eq(true).sum().item()
    acc = safe_div(correct, total)
    micro_f1 = acc

    macro_f1_sum = 0.0
    for cls in range(num_classes):
        pred_pos = pred == cls
        true_pos = true == cls
        tp = (pred_pos & true_pos).sum().item()
        fp = (pred_pos & (~true_pos)).sum().item()
        fn = ((~pred_pos) & true_pos).sum().item()
        denom = (2 * tp) + fp + fn
        class_f1 = safe_div(2 * tp, denom) if denom > 0 else 0.0
        macro_f1_sum += class_f1
    macro_f1 = safe_div(macro_f1_sum, num_classes)
    return acc, micro_f1, macro_f1


def build_neighbor_samplers(data, train_sizes, val_sizes, test_sizes, batch_size, num_workers):
    """Build split-specific NeighborSampler instances.

    Args:
        data (torch_geometric.data.Data): Full transductive graph.
        train_sizes (list[int]): Hop sizes for train split.
        val_sizes (list[int]): Hop sizes for validation split.
        test_sizes (list[int]): Hop sizes for test split.
        batch_size (int): Number of target nodes in each mini-batch.
        num_workers (int): Worker count used by NeighborSampler.

    Returns:
        tuple: Train, validation, and test samplers.
    """
    edge_index_cpu = data.edge_index.cpu()
    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1).cpu()
    val_idx = data.val_mask.nonzero(as_tuple=False).view(-1).cpu()
    test_idx = data.test_mask.nonzero(as_tuple=False).view(-1).cpu()

    train_sampler = NeighborSampler(
        edge_index=edge_index_cpu,
        sizes=train_sizes,
        node_idx=train_idx,
        num_nodes=int(data.num_nodes),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_sampler = NeighborSampler(
        edge_index=edge_index_cpu,
        sizes=val_sizes,
        node_idx=val_idx,
        num_nodes=int(data.num_nodes),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_sampler = NeighborSampler(
        edge_index=edge_index_cpu,
        sizes=test_sizes,
        node_idx=test_idx,
        num_nodes=int(data.num_nodes),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_sampler, val_sampler, test_sampler


def build_neighbor_samplers_lazy(data, train_sizes, val_sizes, test_sizes, batch_size, num_workers, persistent_workers=True, prefetch_factor=2):
    """Build all NeighborSamplers sharing one SparseTensor adjacency.

    The expensive COO→CSR sort runs exactly once during adj_t construction.
    Val and test samplers receive the same adj_t and only allocate a fresh
    arange value tensor (~13 GB for 1.6 B edges) — no second or third sort.
    This prevents the spike that previously OOM-killed eval on papers100M.

    Args:
        data (torch_geometric.data.Data): Full transductive graph.
        train_sizes (list[int]): Hop sizes for train split.
        val_sizes (list[int]): Hop sizes for validation split.
        test_sizes (list[int]): Hop sizes for test split.
        batch_size (int): Number of target nodes in each mini-batch.
        num_workers (int): Worker count used by NeighborSampler.
        persistent_workers (bool): Keep sampler workers alive between epochs.
        prefetch_factor (int): Number of batches loaded in advance by each worker.

    Returns:
        tuple: (train_sampler, val_sampler, test_sampler)
    """
    from torch_sparse import SparseTensor
    from torch.utils.data import DataLoader

    edge_index_cpu = data.edge_index.cpu()
    num_nodes_int = int(data.num_nodes)

    # Build adj_t_base once: sort + transpose (peak ~50 GB for papers100M).
    # No values stored here — each sampler adds its own arange value tensor.
    adj_t_base = SparseTensor(
        row=edge_index_cpu[0],
        col=edge_index_cpu[1],
        sparse_sizes=(num_nodes_int, num_nodes_int),
    ).t()
    adj_t_base.storage.rowptr()

    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1).cpu()
    val_idx = data.val_mask.nonzero(as_tuple=False).view(-1).cpu()
    test_idx = data.test_mask.nonzero(as_tuple=False).view(-1).cpu()

    kwargs = {}
    if num_workers > 0:
        kwargs['persistent_workers'] = persistent_workers
        kwargs['prefetch_factor'] = prefetch_factor

    def _make_sampler(node_idx, sizes, shuffle):
        # Bypass NeighborSampler.__init__ to inject the pre-built adj_t and force
        # is_sparse_tensor=False. This makes sample() yield EdgeIndex namedtuples
        # (not Adj SparseTensor tuples), which is what _unpack_adj expects.
        sampler = object.__new__(NeighborSampler)
        sampler.sizes = sizes
        sampler.return_e_id = True
        sampler.is_sparse_tensor = False
        sampler.__val__ = None
        # Each sampler gets an independent arange value tensor (~13 GB for 1.6 B edges).
        # rowptr is recomputed from the already-sorted structure in O(nnz) — no re-sort.
        value = torch.arange(adj_t_base.nnz())
        adj_t = adj_t_base.set_value(value, layout='coo')
        adj_t.storage.rowptr()
        sampler.adj_t = adj_t
        if node_idx.dtype == torch.bool:
            node_idx = node_idx.nonzero(as_tuple=False).view(-1)
        DataLoader.__init__(
            sampler,
            node_idx.view(-1).tolist(),
            collate_fn=sampler.sample,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs,
        )
        return sampler

    train_sampler = _make_sampler(train_idx, train_sizes, shuffle=True)
    val_sampler = _make_sampler(val_idx, val_sizes, shuffle=False)
    test_sampler = _make_sampler(test_idx, test_sizes, shuffle=False)

    return train_sampler, val_sampler, test_sampler


def get_lazy_sampler(samplers_dict, split):
    """Lazily retrieve or create a sampler from dict.

    Args:
        samplers_dict (dict): Lazy sampler container from build_neighbor_samplers_lazy.
        split (str): One of 'train', 'val', 'test'.

    Returns:
        NeighborSampler: Requested sampler (created on first access).
    """
    if split in ['val', 'test']:
        if split not in samplers_dict:
            cfg = samplers_dict['_config']
            sizes = cfg['val_sizes'] if split == 'val' else cfg['test_sizes']
            node_idx = cfg['val_idx'] if split == 'val' else cfg['test_idx']
            samplers_dict[split] = NeighborSampler(
                edge_index=cfg['edge_index'],
                sizes=sizes,
                node_idx=node_idx,
                num_nodes=cfg['num_nodes'],
                batch_size=cfg['batch_size'],
                shuffle=False,
                num_workers=cfg['num_workers'],
                **cfg.get('kwargs', {}),
            )
    return samplers_dict.get(split)

class NodeClassifier:
    """Train and evaluate node classification for full-batch or sampled mode.

    Args:
        data (torch_geometric.data.Data): Graph data object.
        model (torch.nn.Module): Model instance with full-batch and sampled forward support.
        optimizer (torch.optim.Optimizer): Optimizer used for parameter updates.
        device (torch.device): CUDA device used for training and evaluation.
        num_classes (int): Number of target classes.
        sampling_enabled (bool): Enables neighbor sampling pipeline.
        mode (str): Backend mode, expected values include "gas" and "gar".
        gar_sampling_variant (str): GAR path variant, either "per_batch_csr" or "hybrid_full_cache".
        eval_mode (str): Evaluation strategy for sampling mode:
            "sampled", "full_graph", or "both".
        eval_every_k_epochs (int): Evaluation cadence. If >1, metrics are reused
            on skipped epochs to reduce runtime and memory pressure.
        sampling_debug_enabled (bool): Enables verbose sampled-batch diagnostics.
        sampling_debug_max_batches (int): Maximum number of sampled batches per
            split to print debug diagnostics for.
        sampling_debug_fail_fast (bool): Raises RuntimeError when sampled-batch
            invariants are violated.
        train_sampler: Optional training sampler.
        val_sampler: Optional validation sampler.
        test_sampler: Optional test sampler.
    """

    def __init__(
        self,
        data,
        model,
        optimizer,
        device,
        num_classes,
        sampling_enabled=False,
        mode='gar',
        gar_sampling_variant='per_batch_csr',
        eval_mode='full_graph',
        eval_every_k_epochs=1,
        sampling_debug_enabled=False,
        sampling_debug_max_batches=2,
        sampling_debug_fail_fast=False,
        train_sampler=None,
        val_sampler=None,
        test_sampler=None,
    ):
        self.data = data.to(device)
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.num_classes = int(num_classes)
        self.sampling_enabled = sampling_enabled
        self.mode = mode
        self.gar_sampling_variant = gar_sampling_variant
        self.eval_mode = eval_mode
        self.eval_every_k_epochs = int(eval_every_k_epochs)
        self.sampling_debug_enabled = bool(sampling_debug_enabled)
        self.sampling_debug_max_batches = int(sampling_debug_max_batches)
        self.sampling_debug_fail_fast = bool(sampling_debug_fail_fast)
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.test_sampler = test_sampler
        self._last_eval_metrics = None
        self._sampling_debug_counts = {
            'train': 0,
            'val': 0,
            'test': 0,
        }

        # Support lazy sampler initialization: if samplers_dict is provided,
        # it means we use on-demand sampler creation for val/test.
        self._samplers_dict = None
        if isinstance(train_sampler, dict) and '_config' in train_sampler:
            self._samplers_dict = train_sampler
            self.train_sampler = train_sampler['train']
            self.val_sampler = None
            self.test_sampler = None
    def _sync_cuda(self):
        """Synchronize CUDA stream when running on GPU for precise timings."""
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def _to_device_adjs(self, adjs):
        """Move sampled adjacency objects to target CUDA device.

        Args:
            adjs (list): List of PyG adjacency objects returned by NeighborSampler.

        Returns:
            list: Same adjacency objects placed on configured device.
        """
        return [adj.to(self.device) for adj in adjs]

    @staticmethod
    def _batch_graph_stats(adjs, n_id):
        """Count sampled nodes and edges for logging throughput statistics.

        Args:
            adjs (list): Adjacency objects for all sampled hops.
            n_id (torch.Tensor): Global node ids participating in the sampled subgraph.

        Returns:
            tuple[int, int]: Number of sampled nodes and total sampled edges.
        """
        num_batch_edges = 0
        for adj in adjs:
            if hasattr(adj, 'edge_index'):
                num_batch_edges += int(adj.edge_index.size(1))
            else:
                num_batch_edges += int(adj[0].size(1))
        return int(n_id.size(0)), num_batch_edges

    def _sampled_forward(self, n_id, adjs, split_name='unknown', batch_idx=-1):
        """Run model forward for one sampled subgraph batch.

        Args:
            n_id (torch.Tensor): Global node ids in sampled subgraph.
            adjs (list): Layer-wise adjacency info from NeighborSampler.
            split_name (str): Split name for diagnostics.
            batch_idx (int): Sampled batch index within split.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Device-side node ids and logits.
        """
        adjs = self._to_device_adjs(adjs)
        n_id = n_id.to(self.device)
        x_batch = self.data.x[n_id]

        layer_max_indices = []
        layer_edge_counts = []
        for adj in adjs:
            edge_index = adj.edge_index if hasattr(adj, 'edge_index') else adj[0]
            edge_count = int(edge_index.size(1))
            max_index = int(edge_index.max().item()) if edge_count > 0 else -1
            layer_edge_counts.append(edge_count)
            layer_max_indices.append(max_index)

        emit_sampling_debug = False
        if self.sampling_debug_enabled and split_name in self._sampling_debug_counts:
            if self._sampling_debug_counts[split_name] < self.sampling_debug_max_batches:
                emit_sampling_debug = True
                print(
                    '[SamplingDebug] split={} batch={} n_id_count={} x_batch_rows={} '
                    'layer_edge_counts={} layer_max_indices={}'.format(
                        split_name,
                        batch_idx,
                        int(n_id.numel()),
                        int(x_batch.size(0)),
                        layer_edge_counts,
                        layer_max_indices,
                    )
                )
                self._sampling_debug_counts[split_name] += 1

        if self.sampling_debug_fail_fast:
            for layer_id, max_index in enumerate(layer_max_indices):
                if max_index >= int(x_batch.size(0)):
                    raise RuntimeError(
                        'Sampled edge index out of local node domain: split={} batch={} layer={} '
                        'max_index={} x_batch_rows={}'.format(
                            split_name, batch_idx, layer_id, max_index, int(x_batch.size(0))
                        )
                    )

        out = self.model.forward_sampled(
            x=x_batch,
            adjs=adjs,
            gar_sampling_variant=self.gar_sampling_variant,
            sampling_debug_enabled=emit_sampling_debug,
            sampling_debug_fail_fast=self.sampling_debug_fail_fast,
            sampling_debug_split=split_name,
            sampling_debug_batch=batch_idx,
        )
        return n_id, out

    def train(self):
        """Run one training epoch.

        Returns:
            dict: Loss, timing, and batch-size statistics for logging.
        """
        self._sync_cuda()
        train_start = time.perf_counter()
        self.model.train()

        if not self.sampling_enabled:
            self.optimizer.zero_grad()
            loss = F.nll_loss(self.model(self.data)[self.data.train_mask], self.data.y[self.data.train_mask])
            loss.backward()
            self.optimizer.step()
            avg_loss = loss.item()
            num_batches = 1
            avg_batch_nodes = float(self.data.num_nodes)
            avg_batch_edges = float(self.data.edge_index.size(1))
        else:
            total_loss = 0.0
            num_batches = 0
            total_batch_nodes = 0
            total_batch_edges = 0
            for batch_idx, (batch_size, n_id, adjs) in enumerate(self.train_sampler):
                num_nodes_batch, num_edges_batch = self._batch_graph_stats(adjs, n_id)
                total_batch_nodes += num_nodes_batch
                total_batch_edges += num_edges_batch

                self.optimizer.zero_grad()
                if self.mode == 'gar' and self.gar_sampling_variant == 'hybrid_full_cache':
                    n_id = n_id.to(self.device)
                    logits = self.model(self.data)
                    target_nodes = n_id[:batch_size]
                    loss = F.nll_loss(logits[target_nodes], self.data.y[target_nodes])
                else:
                    n_id, out = self._sampled_forward(
                        n_id=n_id,
                        adjs=adjs,
                        split_name='train',
                        batch_idx=batch_idx,
                    )
                    target_nodes = n_id[:batch_size]
                    loss = F.nll_loss(out[:batch_size], self.data.y[target_nodes])
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = safe_div(total_loss, num_batches)
            avg_batch_nodes = safe_div(total_batch_nodes, num_batches)
            avg_batch_edges = safe_div(total_batch_edges, num_batches)

        self._sync_cuda()
        train_time_sec = time.perf_counter() - train_start
        return {
            'train_loss': avg_loss,
            'train_time_sec': train_time_sec,
            'train_num_batches': num_batches,
            'train_avg_batch_nodes': avg_batch_nodes,
            'train_avg_batch_edges': avg_batch_edges,
        }

    def _evaluate_with_sampler(self, sampler, split_name='unknown'):
        """Evaluate one split with neighbor sampling.

        Args:
            sampler: Split-specific NeighborSampler.
            split_name (str): Split label used for debug traces.

        Returns:
            tuple[float, float, float]: Accuracy, micro-F1, macro-F1.
        """
        pred_chunks = []
        true_chunks = []
        for batch_idx, (batch_size, n_id, adjs) in enumerate(sampler):
            if self.mode == 'gar' and self.gar_sampling_variant == 'hybrid_full_cache':
                n_id = n_id.to(self.device)
                logits = self.model(self.data)
                target_nodes = n_id[:batch_size]
                pred = logits[target_nodes].max(1)[1]
                true = self.data.y[target_nodes]
            else:
                n_id, out = self._sampled_forward(
                    n_id=n_id,
                    adjs=adjs,
                    split_name=split_name,
                    batch_idx=batch_idx,
                )
                pred = out[:batch_size].max(1)[1]
                true = self.data.y[n_id[:batch_size]]
            pred_chunks.append(pred.detach().cpu())
            true_chunks.append(true.detach().cpu())

        if not pred_chunks:
            return 0.0, 0.0, 0.0

        pred_all = torch.cat(pred_chunks, dim=0)
        true_all = torch.cat(true_chunks, dim=0)
        return compute_scores_from_pred_labels(pred_all, true_all, self.num_classes)

    def _evaluate_full_graph(self):
        """Evaluate train/val/test on full graph.

        Returns:
            dict: Split metrics with unsuffixed keys.
        """
        logits = self.model(self.data)
        metrics = {}
        for split_name, mask in [
            ('train', self.data.train_mask),
            ('val', self.data.val_mask),
            ('test', self.data.test_mask),
        ]:
            acc, f1_micro, f1_macro = compute_acc_micro_macro_f1(
                logits=logits,
                labels=self.data.y,
                mask=mask,
                num_classes=self.num_classes,
            )
            metrics[split_name + '_acc'] = acc
            metrics[split_name + '_f1_micro'] = f1_micro
            metrics[split_name + '_f1_macro'] = f1_macro
        return metrics

    def _evaluate_sampled(self):
        """Evaluate train/val/test with sampler-based approximation.

    Lazily initializes val/test samplers if using lazy sampler mode.

        Returns:
            dict: Split metrics with unsuffixed keys.
        """
        metrics = {}
        split_to_sampler = {
            'train': self.train_sampler,
            'val': self.val_sampler,
            'test': self.test_sampler,
        }
        
        # Handle lazy sampler initialization
        if self._samplers_dict is not None:
            split_to_sampler['val'] = get_lazy_sampler(self._samplers_dict, 'val')
            split_to_sampler['test'] = get_lazy_sampler(self._samplers_dict, 'test')
        
        for split_name, sampler in split_to_sampler.items():
            acc, f1_micro, f1_macro = self._evaluate_with_sampler(sampler, split_name=split_name)
            metrics[split_name + '_acc'] = acc
            metrics[split_name + '_f1_micro'] = f1_micro
            metrics[split_name + '_f1_macro'] = f1_macro
        return metrics

    @staticmethod
    def _suffix_metrics(metrics, suffix):
        """Return copy of metrics where score keys are suffixed.

        Args:
            metrics (dict): Metrics with canonical split keys.
            suffix (str): Suffix label, e.g. ``full_graph`` or ``sampled``.

        Returns:
            dict: Suffixed metric keys.
        """
        out = {}
        for key, value in metrics.items():
            if key.endswith('_acc') or key.endswith('_f1_micro') or key.endswith('_f1_macro'):
                out[key + '_' + suffix] = value
        return out

    def test(self, epoch=None):
        """Run one evaluation epoch for train, val, and test splits.

        Args:
            epoch (int, optional): Current epoch index. Used to skip expensive
                evaluation on non-checkpoint epochs.

        Returns:
            dict: Split-wise metrics and total evaluation time.
        """
        if self.eval_every_k_epochs > 1 and epoch is not None and epoch % self.eval_every_k_epochs != 0:
            # Reuse latest metrics on skipped epochs to avoid expensive eval every step.
            if self._last_eval_metrics is not None:
                reused = dict(self._last_eval_metrics)
                reused['eval_time_sec'] = 0.0
                reused['eval_skipped'] = True
                return reused

        self._sync_cuda()
        eval_start = time.perf_counter()
        self.model.eval()

        with torch.no_grad():
            if not self.sampling_enabled:
                metrics = self._evaluate_full_graph()
            else:
                metrics = {}
                sampled_metrics = None
                full_metrics = None

                if self.eval_mode in ['sampled', 'both']:
                    sampled_metrics = self._evaluate_sampled()
                if self.eval_mode in ['full_graph', 'both']:
                    full_metrics = self._evaluate_full_graph()

                # Canonical keys remain unsuffixed for backward-compatible logging.
                # In "both" mode we prefer full-graph metrics as primary quality signal.
                if self.eval_mode == 'sampled':
                    metrics.update(sampled_metrics)
                else:
                    metrics.update(full_metrics)

                if self.eval_mode == 'both':
                    metrics.update(self._suffix_metrics(sampled_metrics, 'sampled'))
                    metrics.update(self._suffix_metrics(full_metrics, 'full_graph'))

        self._sync_cuda()
        metrics['eval_time_sec'] = time.perf_counter() - eval_start
        metrics['eval_skipped'] = False
        self._last_eval_metrics = dict(metrics)
        return metrics
