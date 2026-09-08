import gc
import warnings

import torch


def _patch_ogb_process_meta_fields(dataset_cls):
    """Patch OGB process() to handle NaN optional metadata fields.

    OGB 1.3.2 expects string values in ``meta_info`` for
    ``additional node files`` and ``additional edge files`` and calls
    ``.split(',')`` on them. With some pandas versions, empty CSV cells are
    loaded as ``float('nan')``, which raises ``AttributeError``.

    This patch normalizes those optional fields to ``'None'`` before delegating
    to the original ``process`` implementation.

    Args:
        dataset_cls: ``PygNodePropPredDataset`` class object.
    """
    if getattr(dataset_cls, '_fusegnn_meta_patch_applied', False):
        return

    original_process = dataset_cls.process

    def patched_process(self):
        for key in ['additional node files', 'additional edge files']:
            value = self.meta_info[key]
            if not isinstance(value, str):
                self.meta_info[key] = 'None'
        return original_process(self)

    dataset_cls.process = patched_process
    dataset_cls._fusegnn_meta_patch_applied = True


class OgbNodePropPredDataset:
    """Wrapper for OGB node classification datasets in PyG format.

    This adapter keeps the same interface used by the existing training script:
    ``dataset[0]``, ``dataset.num_features``, and ``dataset.num_classes``.

    Args:
        root (str): Dataset root directory.
        name (str): OGB dataset name, e.g. ``ogbn-arxiv``.

    Raises:
        ImportError: If OGB is not installed in the current environment.
        ValueError: If provided dataset name is not supported by this wrapper.
    """

    SUPPORTED_NAMES = {
        'ogbn-arxiv',
        'ogbn-products',
        'ogbn-papers100M',
    }

    def __init__(self, root, name):
        if name not in self.SUPPORTED_NAMES:
            raise ValueError('Unsupported OGB dataset: {}'.format(name))

        try:
            from ogb.nodeproppred import PygNodePropPredDataset
        except ImportError as exc:
            raise ImportError(
                'OGB is not installed. Install a PyG 1.6.3-compatible version, for example: pip install ogb==1.3.2'
            ) from exc

        _patch_ogb_process_meta_fields(PygNodePropPredDataset)

        if name == 'ogbn-papers100M':
            warnings.warn(
                'ogbn-papers100M is extremely large. Use sampling and run on a high-memory machine.',
                RuntimeWarning,
            )

        self.name = name
        _inner = PygNodePropPredDataset(name=name, root=root)
        self.num_features = int(_inner.num_features)
        self.num_classes = int(_inner.num_classes)

        data = _inner[0]
        split_idx = _inner.get_idx_split()

        # OGB labels are often shape [N, 1]; flatten and cast to int64 for NLL loss.
        data.y = data.y.view(-1).long()

        num_nodes = int(data.num_nodes)
        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        data.train_mask[split_idx['train']] = True
        data.val_mask[split_idx['valid']] = True
        data.test_mask[split_idx['test']] = True

        # Convert features to float16 for papers100M before releasing the inner
        # dataset.  The key ordering constraint: create the independent float16
        # tensor first, replace data.x (drops any view of the float32 storage),
        # then delete _inner so the only remaining float32 reference is gone.
        # Doing it after del _inner would be too late — data.x might still be a
        # view keeping the float32 parent alive.
        if name == 'ogbn-papers100M':
            data.x = data.x.to(torch.float16)

        # Release the OGB/PyG dataset object.  It internally caches tensors
        # (data.x in float32, slices, etc.) that are no longer needed now that
        # we hold our own data object.  Freeing it reclaims ~57 GB for papers100M.
        del _inner
        gc.collect()

        self._data = data

    def __getitem__(self, idx):
        """Return the single transductive graph data object.

        Args:
            idx (int): Graph index. Only ``0`` is valid for OGB node datasets.

        Returns:
            torch_geometric.data.Data: Graph data object with split masks.
        """
        if idx != 0:
            raise IndexError('Only index 0 is available for {}'.format(self.name))
        return self._data

    def __len__(self):
        """Return the number of graphs in the dataset (always 1)."""
        return 1

    def __repr__(self):
        return '{}()'.format(self.name)
