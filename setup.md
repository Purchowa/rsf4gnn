# Local setup for fuseGNN (Windows and WSL2)

This project builds custom CUDA extensions from `src/setup.py` (`fgnn_agg`, `fgnn_format`, `fgnn_gcn`, `fgnn_gat`).

## Recommended path: Ubuntu in WSL2

WSL2 is the most reliable option for this codebase because the original environment and build flow are Linux-like.

1. Install prerequisites in Ubuntu (inside WSL2):

```bash
sudo apt update
sudo apt install -y build-essential python3.8 python3.8-venv python3.8-dev
```

2. Verify GPU passthrough in WSL2:

```bash
nvidia-smi
```

   > **Note:** The "CUDA Version" shown by `nvidia-smi` (e.g. 13.1) is the **maximum** version supported by the Windows host driver — it does not mean CUDA Toolkit 13.1 is installed inside WSL2. PyTorch 1.7.0 requires CUDA Toolkit **11.0** installed separately inside WSL2 (steps 3–4 below). On Colab the toolkit was pre-installed, so this wasn't an issue there.

3. Install CUDA Toolkit 11.0 inside WSL2 (toolkit only — do **not** install the full `cuda` package as it would try to install a driver and break GPU passthrough):

```bash
# Add NVIDIA package repo — adjust distro slug
# Ubuntu 20.04 → ubuntu2004/x86_64  |  Ubuntu 22.04 → ubuntu2204/x86_64
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-11-0
```

4. Set `CUDA_HOME` and update `PATH` / `LD_LIBRARY_PATH` (add to `~/.bashrc` for persistence):

```bash
echo 'export CUDA_HOME=/usr/local/cuda-11.0' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

   Quick check:

```bash
nvcc --version   # must show: Cuda compilation tools, release 11.0
echo $CUDA_HOME  # must be non-empty: /usr/local/cuda-11.0
```

5. Create and activate a Conda environment (necessary for cuda runtime library compatibility if not installed in previous step):

```bash
conda create -n fusegnn38 python=3.8 -y
conda activate fusegnn38
python -m pip install --upgrade pip setuptools wheel
```

6. Install libraries exactly like on Colab:

```bash
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch -y
```

Then install PyG + utilities from pip:

```bash
pip install -r requirements.txt
```

7. Build and install the CUDA extensions:

```bash
cd src
bash install.sh
cd ..
```

8. Quick import test:

```bash
python -c "import torch, torch_scatter, torch_sparse, torch_cluster, torch_spline_conv, torch_geometric; print(torch.__version__, torch.version.cuda)"
python -c "import fgnn_agg, fgnn_format, fgnn_gcn, fgnn_gat; print('fuseGNN CUDA extensions OK')"
```

## Native Windows path (possible, less stable)

Use this only if you want to avoid WSL2.

1. Install:
- Python 3.8 (64-bit)
- Visual Studio Build Tools 2019 or 2022 with C++ workload
- NVIDIA CUDA Toolkit (compatible with your GPU/driver)

2. Create and activate Conda env in PowerShell:

```powershell
conda create -n fusegnn38 python=3.8 -y
conda activate fusegnn38
python -m pip install --upgrade pip setuptools wheel
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch -y
pip install -r requirements.txt
```

3. Build extensions:

```powershell
cd src
python -W ignore setup.py build
python -W ignore setup.py install
cd ..
```

## Notes that matter for compilation

- `src/setup.py` uses `-arch=sm_70`. If your GPU architecture differs, update this flag or set:

```bash
export TORCH_CUDA_ARCH_LIST="7.0"
```

- Keep Python 3.8 for this legacy dependency set.
- Mixing `conda` (for `torch`) and `pip` (for PyG wheels) is acceptable for this project.
- If installation of old wheels fails, run in WSL2 first, then optionally attempt native Windows.
