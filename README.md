# Running Form Analyzer

A real-time computer vision project built with Python, OpenCV, and YOLOv8.

Currently analyzes webcam footage or a local video file, calculates basic pose metrics, and can send a metric summary to a local Ollama model for running-form feedback.

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

## Analyze a Local Video

Replace the video path with your own file:

```bash
uv run webcam.py --source ./my-run-video.mp4 --output ./out/annotated-run.mp4 --json-out ./out/run-metrics.json
```

To preview while it runs:

```bash
uv run webcam.py --source ./my-run-video.mp4 --show
```

To send the measured metrics to your Ollama server at `10.0.0.201`:

```bash
uv run webcam.py --source ./my-run-video.mp4 --ollama --ollama-host 10.0.0.201:11434 --ollama-model llama3.2
```

If your Ollama model has a different name, swap `llama3.2` for whatever `ollama list` shows on that machine.

## Status

Work in progress — live pose estimation is running, moving toward metric calculation and feedback.
