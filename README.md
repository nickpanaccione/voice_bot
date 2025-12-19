# Voice Robot

Voice-controlled iRobot Create3 using wake word detection, speech recognition, and LLM intent parsing.

## Features
- **Wake word**: Say "computer" to activate
- **Movement**: "move forward 2 meters", "turn right 90 degrees"
- **Follow mode**: "follow me" - robot follows you using camera
- **Vision**: "describe your surroundings" - uses Gemini to describe what the robot sees
- **Interrupt**: Say "computer" while moving to stop

## Requirements
- Raspberry Pi 5 with ROS2 Jazzy
- iRobot Create3
- USB webcam with microphone (e.g., Logitech C270)
- Ollama running locally with `gemma3:1b` or larger

## Setup

```bash
# install dependencies
sudo apt install espeak-ng pulseaudio-utils

# create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# install python packages
pip install -r requirements.txt

# copy and edit config
cp .env.example .env
nano .env  # Add API keys
```


## Run

```bash
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
python main.py
```

## Commands
| Command | Action |
|---------|--------|
| "move forward X meters" | Move forward |
| "turn left/right X degrees" | Turn in place |
| "follow me" | Follow human using camera |
| "describe your surroundings" | Describe what robot sees |
| "stop" | Stop all movement |
| "shut down" | Exit program |
