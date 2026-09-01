#!/usr/bin/env bash
# Restores the Vitis AI 3.0 container to a working state for the
# pt_yolox-nano_coco_416_416_1G_3.0 quantization flow.
#
# MUST BE SOURCED, not executed:
#     source /workspace/setup_container.sh
#
# Why this file exists: the Vitis AI container's writable layer is discarded
# on exit. pip installs and any editable install of `yolox` do NOT survive.
# Files under /workspace DO survive (bind mount), so the source patches below
# persist between sessions and are re-applied only if missing.

set -u
PKG=/workspace/model_zoo/pt_yolox-nano_coco_416_416_1G_3.0

if [ ! -d "$PKG" ]; then
    echo "ERROR: package not found at $PKG"
    return 1 2>/dev/null || exit 1
fi

echo "== 1/4  Python dependencies =="
# pycocotools: COCO mAP evaluation. thop: FLOP counter used by model summary.
# protobuf pinned: XIR uses protobuf for xmodel serialization. Do NOT let pip
# move it. Upstream YOLOX's requirements.txt also lists onnx==1.8.1,
# onnxruntime==1.8.0, onnx-simplifier, tensorboard, ninja -- all deliberately
# NOT installed: they are Megvii's export/training deps, would fight the
# container's own ONNX/protobuf stack, and are unused by the Vitis AI flow.
pip install --quiet pycocotools thop protobuf==3.18.1

echo "== 2/4  Package import path =="
# `yolox` is not installed and must not be: `pip install -e code/` would try to
# satisfy the pinned onnx deps above. A path variable achieves the same import
# with zero risk. The compiled fast_cocoeval .so already lives in-tree and
# survives in the bind mount, so no rebuild is needed.
export PYTHONPATH="$PKG/code${PYTHONPATH:+:$PYTHONPATH}"

echo "== 3/4  Source patches (idempotent) =="

# (a) yolox/core/__init__.py eagerly imports Trainer, which imports
#     torch.utils.tensorboard. Quantization and eval never use Trainer.
#     Patched out rather than installing tensorboard, which would pull a
#     protobuf that conflicts with XIR.
F="$PKG/code/yolox/core/__init__.py"
if grep -q '^from \.trainer import Trainer' "$F"; then
    cp "$F" "$F.orig" 2>/dev/null || true
    sed -i 's|^from \.trainer import Trainer|# from .trainer import Trainer  # patched: training-only import pulls tensorboard; unused by quant/eval|' "$F"
    echo "   patched core/__init__.py"
fi

# (b) tools/quant.py hardcodes CUDA for the device and for checkpoint
#     map_location. Guarded behind is_available(). `device` also propagates
#     into torch_quantizer(..., device=device), so this one edit fixes model
#     placement, dummy input, and the quantizer together.
F="$PKG/code/tools/quant.py"
if grep -q "^    device = torch.device('cuda')$" "$F"; then
    cp "$F" "$F.orig" 2>/dev/null || true
    sed -i "s|^    device = torch.device('cuda')$|    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')|" "$F"
    echo "   patched quant.py (device)"
fi
if grep -q '^        loc = "cuda:{}"\.format(rank)$' "$F"; then
    sed -i 's|^        loc = "cuda:{}"\.format(rank)$|        loc = "cuda:{}".format(rank) if torch.cuda.is_available() else "cpu"|' "$F"
    echo "   patched quant.py (map_location)"
fi

# (c) coco_evaluator_q.py hardcodes CUDA tensor types. Same treatment already
#     applied to the non-_q evaluator on Day 2 (see patches/cpu_eval.patch).
F="$PKG/code/yolox/evaluators/coco_evaluator_q.py"
if grep -q '^        tensor_type = torch\.cuda\.HalfTensor if half else torch\.cuda\.FloatTensor$' "$F"; then
    cp "$F" "$F.orig" 2>/dev/null || true
    sed -i 's|^        tensor_type = torch\.cuda\.HalfTensor if half else torch\.cuda\.FloatTensor$|        tensor_type = (torch.cuda.HalfTensor if half else torch.cuda.FloatTensor) if torch.cuda.is_available() else (torch.HalfTensor if half else torch.FloatTensor)|' "$F"
    echo "   patched coco_evaluator_q.py (tensor_type)"
fi
if grep -q '^        statistics = torch\.cuda\.FloatTensor(\[inference_time, nms_time, n_samples\])$' "$F"; then
    sed -i 's|^        statistics = torch\.cuda\.FloatTensor(\[inference_time, nms_time, n_samples\])$|        statistics = (torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)([inference_time, nms_time, n_samples])|' "$F"
    echo "   patched coco_evaluator_q.py (statistics)"
fi

# (d) run_quant.sh requests one GPU; quant.py asserts
#     args.devices <= torch.cuda.device_count(), which is 0 here.
F="$PKG/code/run_quant.sh"
if grep -q '^GPU_NUM=1$' "$F"; then
    cp "$F" "$F.orig" 2>/dev/null || true
    sed -i 's|^GPU_NUM=1$|GPU_NUM=0|' "$F"
    echo "   patched run_quant.sh (GPU_NUM)"
fi

echo "== 4/4  Verify =="
python - << 'PYEOF'
import importlib, sys
ok = True
for m in ("torch", "torchvision", "yolox", "pycocotools", "thop", "pytorch_nndct"):
    try:
        importlib.import_module(m)
        print("   OK   %s" % m)
    except Exception as e:
        print("   FAIL %s -- %s" % (m, e)); ok = False
import torch, google.protobuf as pb
print("   torch %s | protobuf %s | cuda_available=%s"
      % (torch.__version__, pb.__version__, torch.cuda.is_available()))
sys.exit(0 if ok else 1)
PYEOF

echo
echo "Ready. Run the quantization flow from the package root:"
echo "  cd $PKG && bash code/run_quant.sh"
