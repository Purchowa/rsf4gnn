"""
Training GCN/GAT on CiteSeer, Cora, PubMed, and Reddit datasets
"""
import argparse
import os
import random
import signal
import time
import numpy as np
import torch
from fuseGNN.modules import GCN, gcn_config
# from fuseGNN.modules import GAT, gat_config TODO: Fix the GAT implementation and add it to the training script
from fuseGNN.dataloader import Citations, Reddit, OgbNodePropPredDataset
from fuseGNN.training import NodeClassifier, build_neighbor_samplers_lazy, parse_sizes
from fuseGNN.utils import Logger
from fuseGNN.utils import safe_div
from fuseGNN.utils import EnergyMonitor
from fuseGNN.utils.debug_logger import (
    console_log,
    get_current_cpu_memory_mb,
    get_current_gpu_allocated_mb,
    get_current_gpu_memory_mb,
    get_peak_cpu_memory_mb,
    get_peak_gpu_memory_mb,
)
from fuseGNN.reordering import compute_degree_permutation, compute_rcm_permutation, reorder_node_tensors, reorder_edges
import torch_geometric.transforms as T
from tqdm import tqdm
import sys
import traceback

parser = argparse.ArgumentParser()
# dataset config
parser.add_argument('--data', choices=['CiteSeer', 'Cora', 'PubMed', 'Reddit', 'ogbn-arxiv', 'ogbn-products', 'ogbn-papers100M'],
                    default='Cora', help='dataset name')
parser.add_argument('--model', choices=['GCN', 'GAT'], default='GCN', help='GCN model')
parser.add_argument('--data_path', type=str, default='./datasets/', help='the path to datasets')
# training config
parser.add_argument('--max_iter', type=int, default=800, help='maximum training iterations')
parser.add_argument('--gpus', type=str, default='0', help='gpu to use')
parser.add_argument('--seed', type=int, default=42, help='global random seed for reproducible runs')
# logging
parser.add_argument('--log_dir', type=str, default='./log/', help='the path to the logs directory')
parser.add_argument('--log_filename', type=str, default='logs.json', help='log filename')
parser.add_argument('--console_log', action='store_true',
                    help='print key pipeline/debug information to console')
parser.add_argument('--console_log_every_k_epochs', type=int, default=1,
                    help='console logging cadence in epochs (must be positive)')
# energy profiling
parser.add_argument('--energy_profiling', action='store_true',
                    help='collect GPU power/energy metrics via NVML (or nvidia-smi fallback) during training')
parser.add_argument('--energy_sample_interval_sec', type=float, default=0.2,
                    help='power sampling interval in seconds for energy profiling')
parser.add_argument('--energy_idle_baseline_sec', type=float, default=0.0,
                    help='measure idle GPU power for N seconds before training to enable baseline subtraction (0 disables)')
# model configures
# geo: torch_geometric GCNConv baseline; ref: unfused reference implementation based on torch_scatter.
# gas/gar: fused aggregation variants.
parser.add_argument('--mode', choices=['geo', 'ref', 'gas', 'gar'], default='gar',
                    help='convolution backend (geo=PyG baseline, ref=unfused reference, gas/gar=fused)')
parser.add_argument('--flow', choices=['target_to_source', 'source_to_target'], default='target_to_source')
# metric config
parser.add_argument('--target_metric', choices=['test_acc', 'test_f1_micro', 'test_f1_macro', 'val_acc', 'val_f1_micro', 'val_f1_macro'],
                    default='val_acc', help='metric used for time-to-target')
parser.add_argument('--target_value', type=float, default=None, help='target value for time-to-target; disabled when omitted')
parser.add_argument('--baseline_epoch_time_sec', type=float, default=None,
                    help=(
                        'baseline epoch time (in seconds) for amortization calculation. '
                        'This should be the average per-epoch training time WITHOUT reordering (e.g., from a prior run with '
                        '--reordering_method=none). Used to compute amortization_epoch: how many training epochs are needed '
                        'for per-epoch time savings (baseline - current_avg) to offset the preprocessing/reordering overhead. '
                        'E.g., if baseline=10s/epoch, current_avg=9s/epoch, and reordering_time=5s, then amortization_epoch=5. '
                        'If omitted or if current training is not faster than baseline, amortization_epoch will be None.'
                    ))
parser.add_argument('--reordering_time_sec', type=float, default=0.0,
                    help='extra preprocessing/reordering time in seconds')
# integration flags for future dynamic graph pipeline
parser.add_argument('--sampling_enabled', action='store_true', help='future integration flag for neighborhood sampling')
parser.add_argument('--sampling_batch_size', type=int, default=512,
                    help='mini-batch size for sampled target nodes')
parser.add_argument('--sampling_train_sizes', type=str, default='25,10',
                    help='neighbors sampled per hop for train split, comma-separated (e.g. 25,10)')
parser.add_argument('--sampling_val_sizes', type=str, default='50,50',
                    help='neighbors sampled per hop for validation split, comma-separated')
parser.add_argument('--sampling_test_sizes', type=str, default='50,50',
                    help='neighbors sampled per hop for test split, comma-separated')
parser.add_argument('--sampling_num_workers', type=int, default=0,
                    help='number of dataloader workers for NeighborSampler (-1 for auto maximum based on environment)')
parser.add_argument('--sampling_persistent_workers', type=bool, default=True,
                    help='whether to keep sampler workers alive between epochs')
parser.add_argument('--sampling_prefetch_factor', type=int, default=4,
                    help='number of batches loaded in advance by each worker')
parser.add_argument('--sampling_debug_enabled', action='store_true',
                    help='enable detailed sampled-batch diagnostics for GAS/GAR debugging')
parser.add_argument('--sampling_debug_max_batches', type=int, default=2,
                    help='maximum sampled batches per split to print debug diagnostics for')
parser.add_argument('--sampling_debug_fail_fast', action='store_true',
                    help='raise RuntimeError when sampled-batch invariants are violated')
parser.add_argument('--gar_sampling_variant', choices=['per_batch_csr', 'hybrid_full_cache'], default='per_batch_csr',
                    help='GAR sampling variant: per-batch COO->CSR/CSC or full-graph cache control variant')
parser.add_argument('--eval_mode', choices=['sampled', 'full_graph', 'both'], default='full_graph',
                    help='evaluation mode used when sampling is enabled')
parser.add_argument('--eval_every_k_epochs', type=int, default=1,
                    help='run evaluation every k epochs to reduce memory/time overhead')
parser.add_argument('--reordering_method', choices=['none', 'degree_desc', 'rcm'], default='none',
                    help='offline graph preprocessing reordering method (use "none" to disable)')
# parser.add_argument('--dynamic_graph', action='store_true', help='future integration flag for dynamic graph pipeline')

args = parser.parse_args()

# configure CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
assert torch.cuda.is_available(), 'CUDA is not available'
device = torch.device('cuda')

# configure energy profiling
# NVML/nvidia-smi address GPUs by physical index, independent of
# CUDA_VISIBLE_DEVICES, so use the first GPU listed in --gpus.
energy_monitor = None
energy_idle_power_w = None
if args.energy_profiling:
    energy_device_index = int(args.gpus.split(',')[0]) if args.gpus else 0
    energy_monitor = EnergyMonitor(
        device_index=energy_device_index,
        sample_interval_sec=args.energy_sample_interval_sec,
        enabled=True,
    )


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic settings make runs more reproducible across configs.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_global_seed(args.seed)

if args.console_log_every_k_epochs <= 0:
    raise ValueError('console_log_every_k_epochs must be a positive integer.')

reordering_enabled = args.reordering_method != 'none'

if args.sampling_enabled and reordering_enabled:
    print('Warning: sampling_enabled + reordering_method can confound measurements; use with care.')

if args.eval_every_k_epochs <= 0:
    raise ValueError('eval_every_k_epochs must be a positive integer.')

if args.sampling_enabled:
    if args.sampling_num_workers == -1:
        # Pytorch DataLoader allows maximum of os.cpu_count()
        import multiprocessing
        args.sampling_num_workers = multiprocessing.cpu_count()
        print(f"Auto-detected maximum workers: {args.sampling_num_workers}")
    if args.sampling_num_workers > 0:
        print('Warning: sampling_num_workers > 0 may reduce strict reproducibility across runs.')

if args.sampling_debug_max_batches <= 0:
    raise ValueError('sampling_debug_max_batches must be a positive integer.')

sampling_train_sizes = parse_sizes(args.sampling_train_sizes)
sampling_val_sizes = parse_sizes(args.sampling_val_sizes)
sampling_test_sizes = parse_sizes(args.sampling_test_sizes)

def cuda_sync_if_needed():
    if device.type == 'cuda':
        torch.cuda.synchronize(device)

# configure dataset
console_log(args.console_log, 'Starting dataset loading: {}'.format(args.data), include_memory=True, device_=device)
dataset_load_start = time.perf_counter()
path = args.data_path + args.data
try:
    if args.data in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Citations(path, args.data, T.NormalizeFeatures())
    elif args.data in ['Reddit']:
        dataset = Reddit(path)
    elif args.data in ['ogbn-arxiv', 'ogbn-products', 'ogbn-papers100M']:
        dataset = OgbNodePropPredDataset(path, args.data)
    data = dataset[0]
    """
    The data contains 
    edge_index=[2, N(e)], test_mask=[N(v)], train_mask=[N(v)], val_mask=[N(v)], x=[N(v), dim], y=[N(v)], deg=[N(v)]
    The deg is very imbalanced, e.g. [1, 168] for Cora, however, I can still try to ignore it at this very begining
    """
except Exception as e:
    print('The dataset does not exist or is not supported.')
    traceback.print_exc()
    sys.exit()
    
dataset_load_time_sec = time.perf_counter() - dataset_load_start
console_log(args.console_log, 'Dataset loaded in {:.3f}s'.format(dataset_load_time_sec), include_memory=True, device_=device)

data_prep_start = time.perf_counter()
data.train_mask = data.train_mask.to(torch.bool)
data.val_mask = data.val_mask.to(torch.bool)
data.test_mask = data.test_mask.to(torch.bool)
data_prep_time_sec = time.perf_counter() - data_prep_start
console_log(args.console_log, 'Data masks converted to bool in {:.3f}s'.format(data_prep_time_sec), include_memory=True, device_=device)

graph_reordering_time_sec = 0.0
total_reordering_time_sec = float(args.reordering_time_sec)
graph_mode = 'full_graph_static'
if args.sampling_enabled:
    graph_mode = 'neighbor_sampling'
if reordering_enabled:
    console_log(args.console_log, 'Graph reordering enabled: {}'.format(args.reordering_method), include_memory=True, device_=device)
    reorder_start = time.perf_counter()
    if args.reordering_method == 'degree_desc':
        perm, perm_inv = compute_degree_permutation(data.edge_index, int(data.num_nodes))
    elif args.reordering_method == 'rcm':
        perm, perm_inv = compute_rcm_permutation(data.edge_index, int(data.num_nodes))
    else:
        raise ValueError('Unsupported reordering method: {}'.format(args.reordering_method))

    reorder_node_tensors(data, perm)
    edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
    data.edge_index, edge_attr = reorder_edges(data.edge_index, edge_attr, perm_inv, args.flow)
    if edge_attr is not None:
        data.edge_attr = edge_attr
    graph_reordering_time_sec = time.perf_counter() - reorder_start
    total_reordering_time_sec += graph_reordering_time_sec
    graph_mode = 'full_graph_reordered'
    console_log(args.console_log, 'Graph reordering completed in {:.3f}s'.format(graph_reordering_time_sec), include_memory=True, device_=device)

if args.model == 'GCN':
    config = gcn_config[args.data]
    use_cached = (not args.sampling_enabled) or (
        args.sampling_enabled and args.mode == 'gar' and args.gar_sampling_variant == 'hybrid_full_cache'
    )
    model = GCN(num_features=dataset.num_features, hidden=config['hidden'],
                num_classes=dataset.num_classes, cached=use_cached, drop_rate=config['drop_rate'], mode=args.mode, flow=args.flow)
elif args.model == 'GAT':
    raise NotImplementedError('GAT is not integrated in this script yet.')
    # config = gat_config[args.data]
    # model = GAT(num_features=dataset.num_features, hidden=config['hidden'],
    #             num_classes=dataset.num_classes, heads=config['head'], drop_rate=config['drop_rate'], mode=args.mode)

logger = Logger(model_=args.model + '_' + args.mode, data_=args.data, log_dir=args.log_dir, log_filename=args.log_filename)

training_interrupted = False
training_interrupt_signal = None


def _interrupt_training(signum, _frame):
    """Mark the run as interrupted and stop training gracefully.

    Args:
        signum (int): POSIX signal number that interrupted the run.
        _frame: Signal handler frame (unused).
    """
    global training_interrupted
    global training_interrupt_signal
    training_interrupted = True
    training_interrupt_signal = signum
    raise KeyboardInterrupt


signal.signal(signal.SIGINT, _interrupt_training)
signal.signal(signal.SIGTERM, _interrupt_training)

num_nodes = int(data.num_nodes)
num_edges = int(data.num_edges) if data.num_edges is not None else int(data.edge_index.size(1))
num_features = int(dataset.num_features)
num_classes = int(dataset.num_classes)
avg_degree = safe_div(num_edges, num_nodes)

logger.set_meta('task', 'node_classification_multiclass')
logger.set_meta('num_nodes', num_nodes)
logger.set_meta('num_edges', num_edges)
logger.set_meta('num_features', num_features)
logger.set_meta('num_classes', num_classes)
logger.set_meta('avg_degree', avg_degree)
logger.set_meta('num_train_nodes', int(data.train_mask.sum().item()))
logger.set_meta('num_val_nodes', int(data.val_mask.sum().item()))
logger.set_meta('num_test_nodes', int(data.test_mask.sum().item()))
logger.set_meta('sampling_enabled', args.sampling_enabled)
logger.set_meta('seed', args.seed)
logger.set_meta('console_log', args.console_log)
logger.set_meta('console_log_every_k_epochs', args.console_log_every_k_epochs)
logger.set_meta('sampling_batch_size', args.sampling_batch_size if args.sampling_enabled else None)
logger.set_meta('sampling_train_sizes', sampling_train_sizes if args.sampling_enabled else None)
logger.set_meta('sampling_val_sizes', sampling_val_sizes if args.sampling_enabled else None)
logger.set_meta('sampling_test_sizes', sampling_test_sizes if args.sampling_enabled else None)
logger.set_meta('sampling_num_workers', args.sampling_num_workers if args.sampling_enabled else None)
logger.set_meta('sampling_debug_enabled', args.sampling_debug_enabled if args.sampling_enabled else False)
logger.set_meta('sampling_debug_max_batches', args.sampling_debug_max_batches if args.sampling_enabled else None)
logger.set_meta('sampling_debug_fail_fast', args.sampling_debug_fail_fast if args.sampling_enabled else False)
logger.set_meta('gar_sampling_variant', args.gar_sampling_variant if args.sampling_enabled and args.mode == 'gar' else None)
logger.set_meta('eval_mode', args.eval_mode if args.sampling_enabled else 'full_graph')
logger.set_meta('eval_every_k_epochs', args.eval_every_k_epochs)
logger.set_meta('reordering_enabled', reordering_enabled)
logger.set_meta('reordering_method', args.reordering_method if reordering_enabled else None)
logger.set_meta('reordering_applied', reordering_enabled)
# logger.set_meta('is_dynamic_graph', args.dynamic_graph)
logger.set_meta('graph_mode', graph_mode)
logger.set_meta('energy_profiling', args.energy_profiling)
if energy_monitor is not None:
    logger.set_meta('energy_backend', energy_monitor.backend)
    logger.set_meta('energy_source', energy_monitor.energy_source)
    logger.set_meta('energy_sample_interval_sec', args.energy_sample_interval_sec)
    logger.set_meta('energy_gpu_name', energy_monitor.gpu_name)

train_sampler = None
val_sampler = None
test_sampler = None
if args.sampling_enabled:
    console_log(
        args.console_log,
        'Building NeighborSamplers (batch_size={}, train_sizes={}, val_sizes={}, test_sizes={})'.format(
            args.sampling_batch_size, sampling_train_sizes, sampling_val_sizes, sampling_test_sizes
        ),
        include_memory=True,
        device_=device,
    )
    train_sampler, val_sampler, test_sampler = build_neighbor_samplers_lazy(
        data=data,
        train_sizes=sampling_train_sizes,
        val_sizes=sampling_val_sizes,
        test_sizes=sampling_test_sizes,
        batch_size=args.sampling_batch_size,
        num_workers=args.sampling_num_workers,
        persistent_workers=args.sampling_persistent_workers,
        prefetch_factor=args.sampling_prefetch_factor if args.sampling_num_workers > 0 else 2,
    )
    # All three samplers share the same SparseTensor (adj_t built once inside
    # build_neighbor_samplers_lazy).  The original COO edge_index is no longer
    # needed when full-graph evaluation is disabled; freeing it reclaims ~25 GB.
    if args.eval_mode == 'sampled' and args.data == 'ogbn-papers100M':
        import gc
        del data.edge_index
        gc.collect()
        console_log(args.console_log, 'Freed COO edge_index (~25 GB)', include_memory=True, device_=device)
    console_log(args.console_log, 'NeighborSamplers ready', include_memory=True, device_=device)

optimizer = torch.optim.Adam([
    dict(params=model.reg_params, weight_decay=config['weight_decay']),
    dict(params=model.non_reg_params, weight_decay=0.)
], lr=config['lr'])

classifier = NodeClassifier(
    data=data,
    model=model,
    optimizer=optimizer,
    device=device,
    num_classes=num_classes,
    sampling_enabled=args.sampling_enabled,
    mode=args.mode,
    gar_sampling_variant=args.gar_sampling_variant,
    eval_mode=args.eval_mode,
    eval_every_k_epochs=args.eval_every_k_epochs,
    sampling_debug_enabled=args.sampling_debug_enabled,
    sampling_debug_max_batches=args.sampling_debug_max_batches,
    sampling_debug_fail_fast=args.sampling_debug_fail_fast,
    train_sampler=train_sampler,
    val_sampler=val_sampler,
    test_sampler=test_sampler,
)

def _finalize_and_write(interrupted=False):
    """Finalize summaries and write JSON log, optionally marking interruption."""
    if energy_monitor is not None:
        if energy_monitor.available:
            energy_monitor.stop()
            energy_summary = energy_monitor.summary()
            for energy_key, energy_value in energy_summary.items():
                logger.set_summary(energy_key, energy_value)
            if energy_idle_power_w is not None:
                logger.set_summary('idle_power_w', energy_idle_power_w)
                avg_power_w = energy_summary.get('avg_power_w')
                if avg_power_w is not None:
                    logger.set_summary('avg_power_above_idle_w', avg_power_w - energy_idle_power_w)
        else:
            logger.set_summary('energy_available', False)
            logger.set_summary('energy_unavailable_reason', energy_monitor.unavailable_reason)

    avg_epoch_time_sec = safe_div(sum(epoch_times), len(epoch_times)) if epoch_times else 0.0
    amortization_epoch = None
    # Amortization calculation: if a baseline epoch time was provided (e.g., from an earlier non-reordered run),
    # compute how many training epochs are needed before the cumulative per-epoch savings offset the reordering cost.
    # Formula: amortization_epoch = total_reordering_time / per_epoch_savings
    # where per_epoch_savings = baseline_epoch_time - current_avg_epoch_time.
    # Example: if baseline=10s, current=9s, reordering=5s, then savings=1s/epoch and amortization=5 epochs.
    # If current_avg >= baseline (no per-epoch improvement), amortization_epoch remains None (reordering did not help).
    if args.baseline_epoch_time_sec is not None and total_reordering_time_sec > 0:
        per_epoch_savings = args.baseline_epoch_time_sec - avg_epoch_time_sec
        if per_epoch_savings > 0:
            amortization_epoch = total_reordering_time_sec / per_epoch_savings

    logger.set_summary('best_test_acc', best_test_acc)
    logger.set_summary('best_test_f1_micro', best_test_f1_micro)
    logger.set_summary('best_test_f1_macro', best_test_f1_macro)
    logger.set_summary('avg_epoch_time_sec', avg_epoch_time_sec)
    logger.set_summary('time_to_target_metric', args.target_metric)
    logger.set_summary('time_to_target_value', args.target_value)
    logger.set_summary('time_to_target_sec', time_to_target_sec)
    logger.set_summary('target_reached_epoch', target_reached_epoch)
    logger.set_summary('target_reached_metric_value', target_reached_metric_value)
    logger.set_summary('baseline_epoch_time_sec', args.baseline_epoch_time_sec)
    logger.set_summary('amortization_epoch', amortization_epoch)
    logger.set_summary('training_interrupted', bool(interrupted))
    logger.set_summary('training_interrupt_signal', training_interrupt_signal)

    interrupted_suffix = '_interrupted'
    if interrupted and not logger.log_file.endswith(interrupted_suffix + '.json'):
        base_name, ext = os.path.splitext(logger.log_file)
        logger.log_file = base_name + interrupted_suffix + ext

    if interrupted:
        console_log(
            args.console_log,
            'Training interrupted | best_test_acc {:.4f}, best_test_f1_micro {:.4f}, avg_epoch_time_sec {:.4f}'.format(
                best_test_acc, best_test_f1_micro, avg_epoch_time_sec
            ),
            include_memory=True,
            device_=device,
        )
    else:
        console_log(
            args.console_log,
            'Training complete | best_test_acc {:.4f}, best_test_f1_micro {:.4f}, avg_epoch_time_sec {:.4f}'.format(
                best_test_acc, best_test_f1_micro, avg_epoch_time_sec
            ),
            include_memory=True,
            device_=device,
        )

    logger.write()


pbar = tqdm(range(args.max_iter))
best_test_acc = 0.
best_test_f1_micro = 0.
best_test_f1_macro = 0.
epoch_times = []
time_to_target_sec = None
cumulative_time_sec = 0.0
target_reached_epoch = None
target_reached_metric_value = None

logger.set_summary('dataset_load_time_sec', dataset_load_time_sec)
logger.set_summary('data_prep_time_sec', data_prep_time_sec)
logger.set_summary('graph_reordering_time_sec', graph_reordering_time_sec)
logger.set_summary('external_reordering_time_sec', args.reordering_time_sec)
logger.set_summary('reordering_time_sec', total_reordering_time_sec)
logger.set_summary('preprocess_time_sec', dataset_load_time_sec + data_prep_time_sec + total_reordering_time_sec)

# Diagnostics capture is handled by external tooling (e.g., Nsight Compute wrapper).
logger.set_diagnostics_meta('capture_enabled', False)

if energy_monitor is not None:
    if energy_monitor.available:
        if args.energy_idle_baseline_sec > 0:
            console_log(args.console_log, 'Measuring idle GPU power for {:.1f}s'.format(args.energy_idle_baseline_sec), include_memory=False, device_=device)
            energy_idle_power_w = energy_monitor.measure_idle_power_w(args.energy_idle_baseline_sec)
            console_log(args.console_log, 'Idle GPU power: {}'.format(energy_idle_power_w), include_memory=False, device_=device)
        energy_monitor.start()
        console_log(args.console_log, 'Energy profiling started (backend={}, source={}, gpu={})'.format(energy_monitor.backend, energy_monitor.energy_source, energy_monitor.gpu_name), include_memory=False, device_=device)
    else:
        console_log(args.console_log, 'Energy profiling requested but unavailable: {}'.format(energy_monitor.unavailable_reason), include_memory=False, device_=device)

try:
    for epoch in pbar:
        if device.type == 'cuda':
            # Reset peak so peak_* reflects max usage within this epoch only.
            torch.cuda.reset_peak_memory_stats(device)

        cuda_sync_if_needed()
        epoch_start = time.perf_counter()
        energy_snapshot_start = energy_monitor.snapshot() if energy_monitor is not None else None

        config['lr_schedular'].update(epoch, optimizer)
        should_console_log_epoch = (epoch % args.console_log_every_k_epochs == 0)
        if should_console_log_epoch:
            console_log(args.console_log, 'Epoch {} started'.format(epoch), include_memory=True, device_=device)

        train_result = classifier.train()
        eval_result = classifier.test(epoch=epoch)

        cuda_sync_if_needed()
        epoch_time_sec = time.perf_counter() - epoch_start
        cumulative_time_sec += epoch_time_sec
        epoch_times.append(epoch_time_sec)

        epoch_energy_j = None
        epoch_avg_power_w = None
        if energy_snapshot_start is not None:
            energy_snapshot_end = energy_monitor.snapshot()
            if energy_snapshot_end is not None:
                epoch_energy_j = energy_snapshot_end['cumulative_energy_j'] - energy_snapshot_start['cumulative_energy_j']
                epoch_avg_power_w = safe_div(epoch_energy_j, epoch_time_sec)

        train_loss = train_result['train_loss']
        train_time_sec = train_result['train_time_sec']
        train_num_batches = train_result['train_num_batches']
        train_avg_batch_nodes = train_result['train_avg_batch_nodes']
        train_avg_batch_edges = train_result['train_avg_batch_edges']
        eval_time_sec = eval_result['eval_time_sec']
        eval_skipped = bool(eval_result.get('eval_skipped', False))
        # Canonical metrics come from eval_mode policy:
        # - sampled: sampled approximation
        # - full_graph: full-graph inference
        # - both: full-graph as canonical + suffixed auxiliary metrics in logs
        train_acc = eval_result['train_acc']
        val_acc = eval_result['val_acc']
        test_acc = eval_result['test_acc']
        train_f1_micro = eval_result['train_f1_micro']
        val_f1_micro = eval_result['val_f1_micro']
        test_f1_micro = eval_result['test_f1_micro']
        train_f1_macro = eval_result['train_f1_macro']
        val_f1_macro = eval_result['val_f1_macro']
        test_f1_macro = eval_result['test_f1_macro']

        processed_nodes = num_nodes
        processed_edges = num_edges
        if args.sampling_enabled:
            processed_nodes = train_avg_batch_nodes * train_num_batches
            processed_edges = train_avg_batch_edges * train_num_batches
        nodes_per_sec = safe_div(processed_nodes, epoch_time_sec)
        edges_per_sec = safe_div(processed_edges, epoch_time_sec)
        lr_current = float(optimizer.param_groups[0]['lr'])
        # Snapshot memory after train+eval: current_* is instantaneous now;
        # peak_* is the maximum observed since the epoch reset above.
        current_gpu_mem_mb = get_current_gpu_memory_mb(device)
        current_gpu_allocated_mb = get_current_gpu_allocated_mb(device)
        peak_gpu_mem_mb = get_peak_gpu_memory_mb(device)
        current_cpu_mem_mb = get_current_cpu_memory_mb()
        peak_cpu_mem_mb = get_peak_cpu_memory_mb()

        logger.add_scalar('train_loss', train_loss, epoch)
        logger.add_scalar('train_acc', train_acc, epoch)
        logger.add_scalar('val_acc', val_acc, epoch)
        logger.add_scalar('test_acc', test_acc, epoch)
        logger.add_scalar('train_f1_micro', train_f1_micro, epoch)
        logger.add_scalar('val_f1_micro', val_f1_micro, epoch)
        logger.add_scalar('test_f1_micro', test_f1_micro, epoch)
        logger.add_scalar('train_f1_macro', train_f1_macro, epoch)
        logger.add_scalar('val_f1_macro', val_f1_macro, epoch)
        logger.add_scalar('test_f1_macro', test_f1_macro, epoch)
        # Optional metrics are emitted only when eval_mode='both'.
        for aux_key in [
            'train_acc_sampled', 'val_acc_sampled', 'test_acc_sampled',
            'train_f1_micro_sampled', 'val_f1_micro_sampled', 'test_f1_micro_sampled',
            'train_f1_macro_sampled', 'val_f1_macro_sampled', 'test_f1_macro_sampled',
            'train_acc_full_graph', 'val_acc_full_graph', 'test_acc_full_graph',
            'train_f1_micro_full_graph', 'val_f1_micro_full_graph', 'test_f1_micro_full_graph',
            'train_f1_macro_full_graph', 'val_f1_macro_full_graph', 'test_f1_macro_full_graph',
        ]:
            if aux_key in eval_result:
                logger.add_scalar(aux_key, eval_result[aux_key], epoch)
        logger.add_scalar('epoch_time_sec', epoch_time_sec, epoch)
        logger.add_scalar('train_time_sec', train_time_sec, epoch)
        logger.add_scalar('train_num_batches', train_num_batches, epoch)
        logger.add_scalar('train_avg_batch_nodes', train_avg_batch_nodes, epoch)
        logger.add_scalar('train_avg_batch_edges', train_avg_batch_edges, epoch)
        logger.add_scalar('eval_time_sec', eval_time_sec, epoch)
        logger.add_scalar('eval_skipped', 1 if eval_skipped else 0, epoch)
        logger.add_scalar('nodes_per_sec', nodes_per_sec, epoch)
        logger.add_scalar('edges_per_sec', edges_per_sec, epoch)
        logger.add_scalar('learning_rate', lr_current, epoch)
        logger.add_scalar('current_gpu_mem_mb', current_gpu_mem_mb, epoch)
        logger.add_scalar('current_gpu_allocated_mb', current_gpu_allocated_mb, epoch)
        logger.add_scalar('peak_gpu_mem_mb', peak_gpu_mem_mb, epoch)
        logger.add_scalar('current_cpu_mem_mb', current_cpu_mem_mb, epoch)
        logger.add_scalar('peak_cpu_mem_mb', peak_cpu_mem_mb, epoch)
        if epoch_energy_j is not None:
            logger.add_scalar('epoch_energy_j', epoch_energy_j, epoch)
            logger.add_scalar('epoch_avg_power_w', epoch_avg_power_w, epoch)
        logger.log['epoch'].append(epoch)

        if should_console_log_epoch:
            console_log(
                args.console_log,
                'Epoch {} finished | loss {:.4f}, train_acc {:.4f}, val_acc {:.4f}, test_acc {:.4f}, train_batches {}'.format(
                    epoch, train_loss, train_acc, val_acc, test_acc, int(train_num_batches)
                ),
                include_memory=True,
                device_=device,
            )

        if not eval_skipped:
            if test_acc > best_test_acc:
                best_test_acc = test_acc
            if test_f1_micro > best_test_f1_micro:
                best_test_f1_micro = test_f1_micro
            if test_f1_macro > best_test_f1_macro:
                best_test_f1_macro = test_f1_macro

        pbar.set_description(
            'Train ACC: %.3f | Test ACC: %.3f | Test F1(micro): %.3f | Loss: %.3f'
            % (train_acc, best_test_acc, best_test_f1_micro, train_loss)
        )

        should_stop_on_target = False
        if args.target_value is not None and not eval_skipped and time_to_target_sec is None:
            target_metric_value = eval_result.get(args.target_metric)
            if target_metric_value is not None and target_metric_value >= args.target_value:
                time_to_target_sec = cumulative_time_sec
                target_reached_epoch = epoch
                target_reached_metric_value = float(target_metric_value)
                should_stop_on_target = True

        if should_stop_on_target:
            console_log(
                args.console_log,
                'Early stop: reached {}={:.4f} (target {:.4f}) at epoch {}'.format(
                    args.target_metric, target_reached_metric_value, args.target_value, target_reached_epoch
                ),
                include_memory=True,
                device_=device,
            )
            break
except KeyboardInterrupt:
    training_interrupted = True
finally:
    _finalize_and_write(interrupted=training_interrupted)
    if energy_monitor is not None:
        energy_monitor.close()