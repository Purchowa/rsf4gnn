This repository is a modified copy of
[apuaaChen/gcnLib](https://github.com/apuaaChen/gcnLib), the source code of:

> Z. Chen, M. Yan, M. Zhu, L. Deng, G. Li, S. Li, and Y. Xie.
> *fuseGNN: Accelerating Graph Convolutional Neural Network Training on GPGPU.*
> ICCAD 2020. DOI: 10.1145/3400302.3415610

The original repository does not provide a software license. The original
code remains the property of its authors; the original authors are not
responsible for the modifications and extensions in this repository.

## File provenance

### Original files, modified for the thesis

| File | Nature of modification |
|---|---|
| `src/cuda/format_kernel.cu` | Csr2csc algorithm `CSR2CSC_ALG2` → `CSR2CSC_ALG_DEFAULT` (compatibility with newer CUDA toolkit) |
| `fuseGNN/convs/gcn_conv.py` | debug instrumentation, fallback for empty mini-batches in GAR, extended `forward()` for sampled mode |
| `fuseGNN/modules/gcn.py` | `forward_sampled`, `_unpack_adj`, sampled-mode support |
| `fuseGNN/functional/gcn.py` | sampling debug instrumentation, additional validations |
| `fuseGNN/functional/__init__.py`, `convs/__init__.py`, `modules/__init__.py` | export adjustments |
| `fuseGNN/utils/logger.py` | configurable log filename, `meta`/`summary`/`diagnostics` blocks, `set_meta` |
| `fuseGNN/utils/__init__.py`, `dataloader/__init__.py`, `testbench/gcn_conv_tb.py` | adjustments for the above |

### Files added by the thesis author

| File | Purpose |
|---|---|
| `fuseGNN/reordering/ops.py` | degree-descending and RCM node permutations, tensor/edge reindexing |
| `fuseGNN/training/node_classifier.py` | `NodeClassifier`, lazy neighbor samplers, `parse_sizes` |
| `fuseGNN/dataloader/ogb_loader.py` | OGB dataset wrapper (arxiv/products/papers100M) |
| `fuseGNN/utils/energy_monitor.py` | GPU power/energy profiling (NVML / nvidia-smi) |
| `fuseGNN/utils/ncu_parser.py` | Nsight Compute CSV parsing (L2 counters) |
| `fuseGNN/utils/counter_analysis.py` | hardware counter analysis and run comparison |
| `fuseGNN/utils/debug_logger.py` | console debug logging, CPU/GPU memory helpers |
| `training_main.py` | extended training entry point (OGB, sampling, reordering, energy profiling, seeds, TTT) |
| `aggregate_seed_metrics.py`, `plot_compare_metrics.py`, `_plot_compare_metrics/`, `run_ncu_capture.py` | experiment aggregation, plotting, NCU capture |
| `requirements.txt`, `setup.md` | environment specification and setup guide |
````