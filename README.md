# Running Form Analyzer

A real-time computer vision project built with Python, OpenCV, and YOLOv8.

Currently streams webcam footage and runs live pose estimation on each frame, displaying detected keypoints and skeleton overlay.

## Goal

Build a running form analyzer that tracks joint keypoints in real time and calculates biomechanical metrics like knee angle, trunk lean, and cadence to give actionable feedback on running form.

## Stack

- Python 3.11
- OpenCV
- Ultralytics YOLOv8
- PyTorch

## Setup

```bash
uv sync
uv run webcam.py
```

Press `q` to quit.

## Status

Work in progress — live pose estimation is running, moving toward metric calculation and feedback.
