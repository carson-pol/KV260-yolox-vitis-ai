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


## Post-Training Quantization — Day 3

Ran AMD's shipped PTQ flow (`code/run_quant.sh`) end-to-end on CPU: calibration,
quantized evaluation, xmodel export.

INT8 mAP landed at **0.136**, matching both AMD's documented PTQ figure and the
prediction committed to this README at `b219839` on 2026-08-30 — before the
quantization run existed. The prediction was recorded in advance so that a match
would be evidence rather than rationalization.

Metrics below are from the **test pass** (`logs/day3_quant_run.log`, line 1658).
The calibration pass also emits a COCO table, but it is measured while
quantization scales are still being tuned and should not be reported.

| Metric | Float | INT8 (PTQ) | Retained |
|---|---|---|---|
| AP @[.50:.95] | 0.220 | **0.136** | 62% |
| AP @.50 | 0.365 | 0.264 | 72% |
| AP @.75 | 0.226 | 0.132 | 58% |
| AP small | 0.062 | 0.041 | 66% |
| AP medium | 0.225 | 0.155 | 69% |
| AP large | 0.357 | 0.226 | 63% |
| AR @[.50:.95] maxDets=100 | 0.384 | 0.298 | 78% |

Recovering the remaining gap to 0.210 requires quantization-aware training over
full COCO. Out of scope, and declared so before the run rather than after seeing
the number.

### Calibration scales are identical to AMD's reference

The generated `quant_info.json` was diffed against the one AMD ships in the
package's `quantized/` directory:

    diff <(python3 -m json.tool quantize_result/quant_info.json) \
         <(python3 -m json.tool quantized/quant_info.json)

Across 2,025 lines the only difference is a `version` metadata key
(`3.0.0+a44284e+torch1.12.1`) that AMD's 2022 build did not emit. **Every
quantization scale matches exactly.** The mAP agreement is therefore not two
similar procedures coincidentally landing near each other — it is the same
procedure producing the same numbers.

Both files record `"bias_corrected": true`. Bias correction is part of nndct's
default PTQ configuration, not an opt-in step: `run_quant.sh` never passes
`--fast_finetune`, and the quantizer log confirms "Quant config file is empty,
use default quant configuration".

### Where INT8 costs the most

AP@.50 retains 72% of float performance while AP@.75 retains 58%. The model
still finds objects; it places their boxes less precisely. The box regression
head, not detection or classification, is the main casualty of INT8.

Going in, the expectation was that small objects would be the dominant failure
mode — AP-small was the weakest float metric at 0.062. It degraded, but at 66%
retention it held up better than the 62% headline. The prediction was
directionally reasonable and not the largest effect.

### On the calibration set

`run_quant.sh` passes no calibration data argument — only config, checkpoint,
batch size, device count, confidence threshold, quant mode, and output
directory. AMD's reference PTQ flow calibrates on whatever dataloader the
evaluation config builds, which is val2017. The published 0.136 was produced by
calibrating on the evaluation set.

The original plan here was a seeded 500-image calibration draw from train2017,
keeping calibration and evaluation strictly disjoint. That was set aside
deliberately. The goal was validating pipeline correctness, and the only
available reference point is AMD's number; substituting a different calibration
set would have made any mismatch against 0.136 uninterpretable — impossible to
distinguish a broken pipeline from a different experiment. Reproduce first, vary
second.

A disjoint train2017 calibration run remains a worthwhile follow-up if time
allows, with the reproduced baseline as control.

### Environment reconstruction

The Vitis AI container's writable layer is discarded on exit. Files under
`/workspace` survive (bind mount); `pip` installs and any editable install of
`yolox` do not. Day 2's environment had to be rebuilt from scratch, which is why
`scripts/setup_container.sh` exists — container entry is reproducible rather
than remembered.

| Issue | Resolution |
|---|---|
| `yolox` not importable | `PYTHONPATH=<pkg>/code`, not `pip install -e code/` |
| Missing `pycocotools`, `thop` | Installed with `protobuf==3.18.1` pinned explicitly |
| `tensorboard` pulled in by `yolox.core.__init__` | Commented out the training-only `Trainer` import |
| `quant.py` hardcodes CUDA | Two lines guarded behind `torch.cuda.is_available()` |
| `coco_evaluator_q.py` hardcodes CUDA tensor types | Same treatment as `patches/cpu_eval.patch` |
| `GPU_NUM=1` trips the device-count assertion | Set to `0` in `run_quant.sh` |

**`PYTHONPATH` over `pip install -e code/`:** the package's `requires.txt`
inherits upstream YOLOX's pins — `onnx==1.8.1`, `onnxruntime==1.8.0`,
`onnx-simplifier==0.3.5` — which would fight the container's own ONNX and
protobuf stack. Those are Megvii's export dependencies, unused by the Vitis AI
flow. A path variable achieves the same import with no dependency resolution.

**Patching out `tensorboard` rather than installing it:** TensorBoard pulls
protobuf, and XIR uses protobuf for xmodel serialization. Overwriting protobuf
3.18.1 to satisfy a training-only dependency would have surfaced later as an
inscrutable export failure. `quant.py` never references `Trainer`, so the import
is simply removed.

### Artifacts

- `logs/day3_quant_run.log` — full run output, version banner, both COCO tables
- `quant_info.json` — generated quantization scales
- `scripts/setup_container.sh` — reproducible container setup
- `quantize_result/YOLOX_0_int.xmodel` — quantized model, input to `vai_c_xir`
  (excluded by `.gitignore`)
