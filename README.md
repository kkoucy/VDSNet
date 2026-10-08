# Learning to Reorder: Value-Driven State Space Models for Underwater Image Enhancement

Official PyTorch implementation of **VDSNet**, accepted by *IEEE Transactions on Image Processing* (TIP).

Kui Jiang, Yan Luo, Junjun Jiang, Ke Gu, Nan Ma, and Xianming Liu

[[arXiv preprint](https://arxiv.org/abs/2505.01224)] [[Code](https://github.com/kkoucy/RD-UIE)]

VDSNet addresses the mismatch between fixed sequential scanning and the sparse, uneven information distribution of underwater scenes. It combines value-driven reordering scanning, multi-granularity value guidance learning, a Mamba–Conv Mixer, and cross-feature bridges. This repository provides both the full VDSNet model and the wavelet-based lightweight VDSNet-S model.

## Repository layout

```text
VDSNet-release/
├── configs/
│   ├── vdsnet_large.yml
│   └── vdsnet_small.yml
├── models/
│   └── vdsnet.py
├── mamba/                  # bundled Mamba source fallback
├── tools/
│   └── get_flops.py
├── data.py
├── inference.py
├── losses.py
├── train.py
├── environment.yml
└── requirements.txt
```

## Environment

The supplied environment records the currently validated CUDA 11.8 stack.

```bash
mamba env create -f environment.yml
mamba activate vdsnet
```

`mamba-ssm` compiles CUDA extensions during installation. Make sure the local CUDA toolkit is compatible with the PyTorch CUDA version. The experiments in the manuscript were run on an NVIDIA RTX 3090.

### Installing Mamba from this repository

The default environment installs `mamba-ssm` from PyPI. If no compatible wheel is available or that installation fails, a clean Mamba source snapshot is bundled in [`mamba/`](mamba):

```bash
MAMBA_FORCE_BUILD=TRUE pip install --no-build-isolation ./mamba
```

This command builds the selective-scan CUDA extension locally, so `nvcc`, `ninja`, and a CUDA toolkit compatible with the installed PyTorch build are required. No precompiled `.so`, object files, or machine-specific build products are stored in this repository. The bundled Mamba code retains its original Apache-2.0 license in [`mamba/LICENSE`](mamba/LICENSE).

## Data preparation

VDSNet is trained with paired UIEB and LSUI images. Prepare matching input and target filenames as follows, or edit the paths in the YAML files.

```text
datasets/UIEB_LSUI/train/
├── input/
│   ├── 0001.png
│   └── ...
└── target/
    ├── 0001.png
    └── ...

datasets/LSUI/test/
├── input/
└── target/
```

The loader pairs images by identical relative path and filename. Both training configurations use 256 × 256 crops.

## Training

The `batch_size` field in each configuration is the batch size **per GPU**.

Single GPU:

```bash
python train.py --config configs/vdsnet_large.yml
python train.py --config configs/vdsnet_small.yml
```

Multi-GPU DistributedDataParallel training:

```bash
torchrun --standalone --nproc_per_node=4 train.py \
  --config configs/vdsnet_large.yml

torchrun --standalone --nproc_per_node=4 train.py \
  --config configs/vdsnet_small.yml
```

MVGL uses a frozen local DINOv2-Base model during training only. Download a Hugging Face-compatible DINOv2-Base directory and place it at `pretrained_models/dinov2-base`, or update `dino_model_path` in the configuration. DINOv2 is excluded from inference and from saved restoration checkpoints.

Resume training by setting `train.resume` to a saved checkpoint. To initialize only the restoration network, set `train.pretrained` instead.

## Inference

Inference accepts either one image or a directory. Directory traversal is recursive and preserves relative paths.

```bash
python inference.py \
  --config configs/vdsnet_large.yml \
  --weights /path/to/vdsnet.pth \
  --input /path/to/input_images \
  --output results/VDSNet
```

For VDSNet-S, replace the configuration and checkpoint paths with their Small counterparts.

## Parameters and FLOPs

Run the same THOP-based counting convention for both variants:

```bash
python tools/get_flops.py --config configs/vdsnet_large.yml --height 256 --width 256
python tools/get_flops.py --config configs/vdsnet_small.yml --height 256 --width 256
```

The script prints the executed-graph parameter count, all registered parameters, MACs, and FLOPs under the `1 MAC = 2 FLOPs` convention. The manuscript's complexity table follows the common THOP convention of labeling the MAC count as GFLOPs. Generic THOP hooks do not fully count sorting/indexing or a custom fused selective-scan kernel, so keep the script and software stack fixed when comparing methods.

## Citation

If this work is useful to your research, please cite it as follows. Volume, pages, and DOI will be added when they become available.

```bibtex
@article{jiang2026vdsnet,
  title   = {Learning to Reorder: Value-Driven State Space Models for Underwater Image Enhancement},
  author  = {Jiang, Kui and Luo, Yan and Jiang, Junjun and Gu, Ke and Ma, Nan and Liu, Xianming},
  journal = {IEEE Transactions on Image Processing},
  year    = {2026}
}
```

## Contact

For questions about the code or paper, please contact Yan Luo at [luoyan1007@126.com](mailto:luoyan1007@126.com).

## Acknowledgements

This implementation uses the selective-scan operator from [Mamba](https://github.com/state-spaces/mamba). We thank the authors of Mamba, DINOv2, UIEB, and LSUI for their work.
