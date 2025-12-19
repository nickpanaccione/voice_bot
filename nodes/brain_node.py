#!/usr/bin/env python3
import threading
import random
import time

import pvporcupine
from pvrecorder import PvRecorder

from robot import config
from robot.llm_intent import LLMIntentParser
from robot.interfaces import Create3RobotInterface
from robot.speech import WhisperRecognizer
from robot.intent import parse_intent


THINKING_LINES = [
    "Okay, let me look around.",
    "Give me a moment to look around.",
    "One second, I'm analyzing what I see.",
]


# text to speech movement summary
def format_actions_for_speech(actions):
    parts = []

    for action in actions:
        atype = action.get("type")

        if atype == "move":
            direction = action.get("direction", "forward")
            dist = action.get("distance_m")
            if dist is not None:
                if float(dist).is_integer():
                    dist_str = str(int(dist))
                else:
                    dist_str = f"{dist:g}"
                parts.append(f"move {direction} {dist_str} meters")
            else:
                parts.append(f"move {direction}")

        elif atype == "turn":
            direction = action.get("direction", "right")
            deg = action.get("degrees")
            if deg is not None:
                if float(deg).is_integer():
                    deg_str = str(int(deg))
                else:
                    deg_str = f"{deg:g}"
                parts.append(f"turn {direction} {deg_str} degrees")
            else:
                parts.append(f"turn {direction}")

        elif atype == "stop":
            parts.append("stop")

    if not parts:
        return "executing your movement sequence"

    if len(parts) == 1:
        phrase = parts[0]
        if phrase.startswith("move "):
            phrase = "moving " + phrase[len("move "):]
        elif phrase.startswith("turn "):
            phrase = "turning " + phrase[len("turn "):]
        elif phrase == "stop":
            phrase = "stopping"
        return phrase

    if len(parts) == 2:
        first, second = parts
        if first.startswith("move "):
            first = "moving " + first[len("move "):]
        elif first.startswith("turn "):
            first = "turning " + first[len("turn "):]
        return f"{first} and then {second}"

    return ", then ".join(parts)

# command handling
def handle_command(robot, command_text: str, llm_parser) -> str:
    intent = None

    # try LLM parser
    if llm_parser is not None:
        try:
            intent = llm_parser.parse(command_text)
            print(f"[DEBUG] LLM intent: {intent}")
        except Exception as e:
            print(f"[NLU] LLM parsing failed: {e}. Falling back to rule-based parser.")

    # fallback
    if intent is None:
        intent = parse_intent(command_text)
        print(f"[DEBUG] Parsed intent (rule-based): {intent}")
    elif intent.get("intent") == "unknown":
        rb_intent = parse_intent(command_text)
        print(f"[DEBUG] Rule-based backup intent: {rb_intent}")
        if rb_intent.get("intent") != "unknown":
            intent = rb_intent

    kind = intent.get("intent", "unknown")
    print(f"[DEBUG] Final intent: {intent}")

    if kind == "movement_sequence":
        actions = intent.get("actions", [])
        if not actions:
            return "unknown"

        summary = format_actions_for_speech(actions)
        robot.speak(f"Okay, {summary}.")
        
        # movement in background thread
        def move_worker():
            robot.execute_movement_sequence(actions)
        
        threading.Thread(target=move_worker, daemon=True).start()
        return "movement_started"

    elif kind == "follow":
        # start following
        robot.speak("Okay, I'll follow you. Say computer to stop.")
        if robot.start_following():
            return "following_started"
        else:
            robot.speak("Sorry, I couldn't start following. Camera may not be available.")
            return "recognized"

    elif kind == "describe_surroundings":
        result = {}
        done_event = threading.Event()

        def worker():
            print("[DESC] Worker: starting describe_scene()")
            try:
                desc = robot.describe_scene()
                result["description"] = desc
                print("[DESC] Worker: got description.")
            except Exception as e:
                print(f"[DESC] Worker error: {e}")
                result["description"] = "I'm having trouble seeing right now."
            finally:
                done_event.set()

        t = threading.Thread(target=worker, daemon=True)
        t.start()

        thinking_line = random.choice(THINKING_LINES)
        robot.speak(thinking_line)

        if not done_event.is_set():
            print("[DESC] Main: waiting for vision...")
            robot.play_processing_start()
            done_event.wait()
            robot.play_processing_stop()

        description = result.get("description", "I'm not sure what I see.")
        print(f"[DESC] Main: final description: {description!r}")
        robot.speak(description)
        return "recognized"

    elif kind == "stop":
        robot.speak("Stopping.")
        robot.stop()
        return "stop"

    elif kind == "follow_me":
        robot.speak("Okay, I'll follow you. Say computer to stop.")
        robot.start_following()
        return "following_started"

    elif kind == "shutdown":
        robot.speak("Shutting down. Goodbye.")
        robot.play_sleep_sound()
        return "shutdown"

    elif kind == "help":
        robot.speak(
            "I can move forward and backward, turn left and right, "
            "follow you, stop, describe my surroundings, and shut down. "
            "For example, say: move forward two meters, follow me, "
            "or describe your surroundings."
        )
        return "recognized"

    return "unknown"

# handle commands
def run_command_session(robot, recognizer, llm_parser):
    attempt = 0

    while True:
        attempt += 1
        text = recognizer.listen_and_transcribe()

        if not text:
            robot.speak("I didn't hear a command.")
            robot.play_sleep_sound()
            return None

        result = handle_command(robot, text, llm_parser)

        if result == "shutdown":
            return "shutdown"
        
        if result == "movement_started":
            # movement started
            return "moving"
        
        if result == "following_started":
            # following running
            return "following"
        
        if result in ("recognized", "stop"):
            robot.play_sleep_sound()
            return None

        # unknown try again
        if attempt == 1:
            robot.speak(
                "I didn't quite catch that. "
                "You can ask me to move, follow you, "
                "or describe your surroundings. Please try again."
            )
        else:
            robot.speak("I'm still not sure what you meant. Going back to sleep.")
            robot.play_sleep_sound()
            return None

# main loop 
def main():
    if not config.PICOVOICE_ACCESS_KEY:
        print("[ERROR] PICOVOICE_ACCESS_KEY not set in .env!")
        return

    print(f"[CONFIG] Mic device index: {config.MIC_DEVICE_INDEX}")
    print(f"[CONFIG] Camera device: {config.CAMERA_DEVICE}")
    print(f"[CONFIG] Ollama: {config.OLLAMA_URL} model={config.OLLAMA_MODEL}")

    robot = Create3RobotInterface()
    device_index = config.MIC_DEVICE_INDEX

    devices = PvRecorder.get_available_devices()
    print("\nAvailable audio devices:")
    for i, name in enumerate(devices):
        marker = " <-- SELECTED" if i == device_index else ""
        print(f"  {i}: {name}{marker}")

    porcupine = pvporcupine.create(
        access_key=config.PICOVOICE_ACCESS_KEY,
        keywords=["computer"],
    )
    recorder = PvRecorder(device_index=device_index, frame_length=porcupine.frame_length)

    recognizer = WhisperRecognizer()

    llm_parser = None
    if config.USE_LLM_INTENT:
        try:
            llm_parser = LLMIntentParser()
            print(f"[NLU] Using Ollama model: {llm_parser.model} at {llm_parser.url}")
        except Exception as e:
            print(f"[NLU] Could not initialize LLMIntentParser: {e}")

    print("\n" + "=" * 50)
    print("Robot is ready! Listening for wake word 'computer'.")
    print("Say 'computer' to wake, or 'computer stop' while moving.")
    print("Ctrl+C to exit.")
    print("=" * 50 + "\n")

    try:
        recorder.start()
        while True:
            pcm = recorder.read()
            result = porcupine.process(pcm)

            if result >= 0:
                # wake word detected
                
                # interrupt if moving 
                if robot.is_moving():
                    if robot.is_following():
                        print("\n[WAKE] Wake word detected - STOPPING FOLLOW!")
                        robot.stop()
                        robot.speak("Stopped following.")
                    else:
                        print("\n[WAKE] Wake word detected - STOPPING MOVEMENT!")
                        robot.stop()
                        robot.speak("Stopped.")
                    robot.play_sleep_sound()
                    print("[IDLE] Listening for wake word...\n")
                    continue
                
                # start command session
                print("\n[WAKE] Wake word detected!")
                recorder.stop()
                recorder.delete()
                robot.play_ack_sound()

                session_result = run_command_session(robot, recognizer, llm_parser)

                if session_result == "shutdown":
                    break
                
                time.sleep(0.5)

                # create new recorder
                print("\n[IDLE] Listening for wake word 'computer'...")
                recorder = PvRecorder(device_index=device_index, frame_length=porcupine.frame_length)
                recorder.start()
                
                if session_result == "moving":
                    print("[INFO] Movement in progress - say 'computer' to stop.\n")
                elif session_result == "following":
                    print("[INFO] Following you - say 'computer' to stop.\n")
                else:
                    print("")

    except KeyboardInterrupt:
        print("\n\nExiting...")
    except SystemExit:
        print("\nShutdown requested.")
    finally:
        try:
            recorder.stop()
            recorder.delete()
        except Exception:
            pass
        porcupine.delete()
        robot.shutdown()

if __name__ == "__main__":
    main()
