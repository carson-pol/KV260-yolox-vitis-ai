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


## Float (FP32) baseline — Day 2

Model: `pt_yolox-nano_coco_416_416_1G_3.0` (Vitis AI 3.0 model zoo)
Eval set: full COCO val2017, 5000 images
Host: CPU-only container, batch 32, `--conf 0.001`

| Metric | Result | AMD published |
| - | - | - |
| AP @[.50:.95] | 0.220 | 0.220 |
| AP @.50 | 0.365 | |
| AP @.75 | 0.226 | |
| AP small | 0.062 | |
| AP medium | 0.225 | |
| AP large | 0.357 | |
| AR @[.50:.95] maxDets=100 | 0.384 | |

Average forward time 8.42 ms (CPU, batch 32). Full log: `results/day2_float_eval_full_val2017.log`

Baseline matches the model zoo's published figure exactly, confirming dataset paths,
letterbox preprocessing (BGR, long side 416, pad (114,114,114)), and evaluator setup.

### CPU patch to model zoo eval tooling

The zoo's `code/tools/eval.py` has no CPU code path: it asserts `num_gpu <=
torch.cuda.device_count()`, then calls `torch.cuda.set_device()` unconditionally.
`code/run_eval.sh` hardcodes `GPU_NUM=1`. On the CPU-only container this fails at
`AttributeError: module 'torch._C' has no attribute '_cuda_setDevice'`.

Patched four CUDA calls behind `torch.cuda.is_available()` guards across two files
(device selection, checkpoint `map_location`, evaluator input tensor type, statistics
tensor type). The TensorRT branch's `.cuda()` call was left untouched as unreachable
without `--trt`. See `patches/cpu_eval.patch`.

### Expected quantization target

AMD's package README documents this model's own results: float 0.220, PTQ 0.136,
QAT 0.210. The published 0.21 INT8 figure requires quantization-aware training over
full COCO, which is out of scope here. **Post-training quantization on this
architecture is expected to land near 0.136**, and that is the target for the INT8
stage rather than 0.21.
