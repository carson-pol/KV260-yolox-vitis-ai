# YOLOX-Nano on Kria KV260 via Vitis AI 3.0

Quantized INT8 object detection on the DPUCZDX8G B4096 DPU.
Status: in progress.

## Toolchain (pinned)
- Vitis AI 3.0, image `xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.0.0.106`
- Python 3.7.12, PyTorch 1.12.1 (CPU), torchvision 0.13.1+cpu
- pytorch_nndct 3.0.0, conda env `vitis-ai-pytorch`
- Host: Windows 11 + WSL2 (Ubuntu 22.04), CPU-only flow

## Pipeline validation — ResNet18 (Day 1)
Float:  69.972 / 88.7586 (top-1 / top-5)
INT8:   69.1308 / 88.8096
Compiled: 3 device subgraphs, 1 DPU subgraph, 171 ops
