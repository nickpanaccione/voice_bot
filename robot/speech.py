#!/usr/bin/env python3
import time
import numpy as np
import whisper
from pvrecorder import PvRecorder

from robot import config


class WhisperRecognizer:
    def __init__(self):
        self.sample_rate = 16000
        self.device_index = config.MIC_DEVICE_INDEX
        self.max_seconds = config.MAX_RECORD_SECONDS
        self.silence_seconds = config.SILENCE_TIMEOUT
        self.energy_threshold = config.ENERGY_THRESHOLD

        print(f"[ASR] Loading Whisper model '{config.WHISPER_MODEL}'...")
        self.model = whisper.load_model(config.WHISPER_MODEL)
        print("[ASR] Whisper model loaded.")

    def listen_and_transcribe(self) -> str:
        frame_length = 512
        recorder = PvRecorder(device_index=self.device_index, frame_length=frame_length)

        print(f"[ASR] Listening (max {self.max_seconds}s, stops on {self.silence_seconds}s silence)...")
        frames = []

        voice_started = False
        last_voice_time = None

        try:
            recorder.start()
            start = time.time()

            while True:
                now = time.time()
                elapsed = now - start

                if elapsed > self.max_seconds:
                    print(f"[ASR] Max recording time ({self.max_seconds}s) reached.")
                    break

                pcm = recorder.read()
                frames.append(pcm)

                # calculate rms
                audio_int16 = np.array(pcm, dtype=np.int16)
                rms = np.sqrt(np.mean(audio_int16.astype(np.float32) ** 2))

                if rms > self.energy_threshold:
                    if not voice_started:
                        print("[ASR] Voice detected...")
                        voice_started = True
                    last_voice_time = now
                else:
                    # check for silence after speech
                    if voice_started and last_voice_time is not None:
                        if now - last_voice_time >= self.silence_seconds:
                            print(f"[ASR] Silence detected after speech, stopping ({elapsed:.1f}s total).")
                            break

        finally:
            recorder.stop()
            recorder.delete()
            time.sleep(0.3)

        # play sleep sound
        from robot.interfaces import _play_sound
        from pathlib import Path
        BASE_DIR = Path(__file__).resolve().parent.parent
        _play_sound(BASE_DIR / "data" / "audio" / "sleep.wav")

        if not frames:
            print("[ASR] No audio captured.")
            return ""

        # convert to float32 for whisper
        audio_int16_all = np.array(frames, dtype=np.int16).flatten()
        audio_float32 = audio_int16_all.astype(np.float32) / 32768.0

        print("[ASR] Running Whisper transcription...")
        result = self.model.transcribe(audio_float32, fp16=False, language="en")
        text = result.get("text", "").strip()
        print(f"[ASR] Transcript: \"{text}\"")
        return text
