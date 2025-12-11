# voice_bot_docker

Lightweight voice-bot project packaged with a Dockerfile.

This repository contains small node-like Python components for a voice bot (ASR, NLU, TTS, motion) and a packaged speech model used by the nodes.

## Repository layout

- `Dockerfile` — Dockerfile to build an image for the project.
- `models/` — pre-downloaded model files (e.g. `en_US-amy-medium.onnx` and its json).
- `data/` — project data (audio samples, configs, etc.).
- `nodes/` — Python node scripts:
  - `asr_node.py` — automatic speech recognition node
  - `nlu_node.py` — natural language understanding node
  - `tts_node.py` — text-to-speech node
  - `motion_node.py` — motion/actuator node

## Purpose

This repo is intended to run the voice-bot nodes either inside Docker or directly on the host for development.

## Quick start — build the Docker image

From the repository root (macOS / zsh):

```bash
docker build -t voice-bot .
```

Run the container (example mounting the `models/` and `data/` folders so changes persist):

```bash
docker run --rm -it \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/data:/app/data" \
  --name voice-bot \
  voice-bot
```

Adjust volumes/ports/env as needed for your container runtime.

## Run nodes locally (for development)

If you prefer to run node scripts directly on the host (recommended during development), ensure you have Python installed and the required packages available in your environment. From the repository root:

```bash
# run each node in a separate terminal
python3 nodes/asr_node.py
python3 nodes/nlu_node.py
python3 nodes/tts_node.py
python3 nodes/motion_node.py
```

Note: The repository currently includes model files in `models/`. If a node requires additional dependencies, install them in a virtualenv or use the Docker image.

## Files of interest

- `models/en_US-amy-medium.onnx` — ONNX model used by TTS/ASR nodes (already included).
- `nodes/*.py` — node entrypoints.
