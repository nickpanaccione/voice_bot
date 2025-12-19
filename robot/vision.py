#!/usr/bin/env python3
import cv2
import google.generativeai as genai

from robot import config

# configure Gemini
if config.GOOGLE_API_KEY:
    genai.configure(api_key=config.GOOGLE_API_KEY)
else:
    print("[VISION WARN] GOOGLE_API_KEY not set - vision will not work")


def capture_single_frame(warmup_frames: int = 10):
    # use device from config
    cap = cv2.VideoCapture(config.CAMERA_DEVICE)
    if not cap.isOpened():
        # fallback
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera at {config.CAMERA_DEVICE}")

    try:
        last_good = None

        for i in range(warmup_frames):
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            last_good = frame

            # check brightness
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if gray.mean() > 20:
                break

        if last_good is None:
            raise RuntimeError("Failed to capture valid frame")

        ok, buf = cv2.imencode(".jpg", last_good)
        if not ok:
            raise RuntimeError("Failed to encode frame as JPEG")

        return buf.tobytes()

    finally:
        cap.release()


def describe_scene_with_gemini() -> str:
    if not config.GOOGLE_API_KEY:
        return "I can't see right now - no API key configured."

    image_bytes = capture_single_frame()

    model = genai.GenerativeModel(config.GEMINI_VISION_MODEL)

    prompt = (
        "You are describing what a small mobile robot sees through its camera.\n"
        "Describe the scene in 2-3 concise sentences.\n"
        "Mention key objects and their positions relative to the robot "
        "(e.g., 'a chair in front', 'a wall to the left').\n"
        "Be conversational, as if talking to the robot's owner."
    )

    print("[VISION] Sending image to Gemini...")
    response = model.generate_content(
        [
            prompt,
            {"mime_type": "image/jpeg", "data": image_bytes},
        ],
        request_options={"timeout": 30},
    )
    print("[VISION] Got response from Gemini.")

    text = (response.text or "").strip()
    if not text:
        return "I'm not sure what I see."

    return text
