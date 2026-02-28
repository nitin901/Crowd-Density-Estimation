# Crowd Density Estimation (Spatial + Temporal)

This repository now contains a **real-time deep learning architecture** for crowd density estimation that addresses:

1. **Temporal analysis of video streams in real time**.
2. **Optimized spatial analysis for crowd density estimation**.

## Proposed Models

### 1) `TemporalSpatialCrowdNet`
A two-stage model:

- **Spatial encoder**: MobileNetV3-Small backbone (lightweight for real-time).
- **Temporal module**: ConvLSTM over frame-level feature maps to model motion and temporal crowd evolution.
- **Density decoder head**: Produces a per-pixel density map for each frame.

This model is suitable for online video processing with a sliding window of frames.

### 2) `OptimizedSpatialCrowdNet`
A lightweight purely spatial network:

- Depthwise-separable convolutions for efficiency.
- Multi-scale feature fusion for better localization.
- Real-time friendly upsampling decoder.

Use this when only single-frame estimation is needed.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python demo.py
```

## Input/Output Shapes

### Temporal model
- Input: `(B, T, 3, H, W)`
- Output density maps: `(B, T, 1, H, W)`

### Spatial model
- Input: `(B, 3, H, W)`
- Output density maps: `(B, 1, H, W)`

## Training Notes

A typical objective is a weighted combination of:

- **Pixel-wise MSE** between predicted and ground-truth density maps.
- **Count loss** between integrated predicted count and true count.

```text
L = λ1 * MSE(D_pred, D_gt) + λ2 * |sum(D_pred) - sum(D_gt)|
```

Use temporal clips for `TemporalSpatialCrowdNet` and single images for `OptimizedSpatialCrowdNet`.

## Real-Time Deployment Suggestions

- Use mixed precision (`torch.cuda.amp.autocast`).
- Export with TorchScript or ONNX.
- Use clip length `T=4` or `T=8` depending on hardware latency budget.
- Keep input resolution at deployment-specific targets (e.g., 512x512, 640x360).

