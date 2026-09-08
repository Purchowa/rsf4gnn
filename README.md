# RSF4GNN

**Reorder-Sample-Fuse for GNN** - extensions for fused GNN training:
neighborhood sampling, graph reordering, and energy profiling, built on top of
[fuseGNN](https://github.com/apuaaChen/gcnLib).

This repository accompanies the MSc thesis *"Analysis and Enhancement of Training Methods for Graph Neural Networks on Large-Scale Graphs"*
(AGH University of Cracow, 2026; author: Dawid Wołek).

## Credits / Attribution

The base code of this repository comes from
[apuaaChen/gcnLib](https://github.com/apuaaChen/gcnLib), the reference
implementation of the paper:

> Z. Chen, M. Yan, M. Zhu, L. Deng, G. Li, S. Li, and Y. Xie.
> *fuseGNN: Accelerating Graph Convolutional Neural Network Training on GPGPU.*
> In Proceedings of the 39th International Conference on Computer-Aided
> Design (ICCAD 2020), 2020.

```bibtex
@inproceedings{chen2020fusegnn,
  author    = {Chen, Zhaodong and Yan, Mingyu and Zhu, Maohua and
               Deng, Lei and Li, Guoqi and Li, Shuangchen and Xie, Yuan},
  title     = {fuseGNN: Accelerating Graph Convolutional Neural Network
               Training on GPGPU},
  booktitle = {Proceedings of the 39th International Conference on
               Computer-Aided Design (ICCAD)},
  year      = {2020}
}
```

This repository is a modified copy of the original code, extended for the
thesis research. The original authors are not responsible for the extensions.
See [NOTICE.md](NOTICE.md) for a file-by-file breakdown of what was changed
and what was added.

## What this fork adds

The original fuseGNN supports full-graph, transductive training only. This
repository (RSF4GNN) adds:

- **Neighborhood sampling integration** - PyTorch Geometric `NeighborSampler`
  integrated with both fused aggregation engines (GAS and GAR), including a
  *per-batch CSR* conversion path for GAR.
- **Graph reordering as preprocessing** - degree-descending sorting and
  Reverse Cuthill-McKee (RCM) node permutation, with edge reindexing and
  re-sorting, applied once on CPU before training.
- **OGB datasets** - `ogbn-arxiv`, `ogbn-products`, `ogbn-papers100M` loaders
  (`fuseGNN/dataloader/ogb_loader.py`).
- **Experimental infrastructure** - unified per-epoch JSON logging (time,
  throughput, VRAM/RAM, accuracy), GPU energy/power profiling via NVML
  (`EnergyMonitor`), Nsight Compute L2 counter capture and parsing,
  multi-seed aggregation (`aggregate_seed_metrics.py`) and comparison plots
  (`plot_compare_metrics.py`).

## Setup

See [setup.md](setup.md) for the full environment setup (WSL2 / Ubuntu,
CUDA Toolkit 11.0, conda `fusegnn38`, PyTorch 1.7.0 + PyG 1.6.3, building the
CUDA extensions from `src/`).

Quick version:

```bash
conda create -n fusegnn38 python=3.8 -y
conda activate fusegnn38
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch -y
pip install -r requirements.txt
cd src && bash install.sh && cd ..
```