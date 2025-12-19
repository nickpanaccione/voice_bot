#!/usr/bin/env python3
import re
from typing import Dict, List

# Word to number mapping
WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "ninety": 90, "hundred": 100,
}

def words_to_number(text: str) -> str:
    result = text.lower()
    for word, num in WORD_NUMBERS.items():
        result = re.sub(rf'\b{word}\b', str(num), result)
    return result


def parse_intent(text: str) -> Dict:
    t = text.lower().strip()
    t = words_to_number(t)

    if not t:
        return {"intent": "unknown"}

    # shutdown
    if any(kw in t for kw in ["shut down", "shutdown", "power off", "turn off"]):
        return {"intent": "shutdown"}

    # help
    if any(kw in t for kw in ["help", "what can you do", "what are you able to do"]):
        return {"intent": "help"}

    # stop
    if t.strip() == "stop" or "stop moving" in t or "stop following" in t:
        return {"intent": "stop"}

    # follow me
    if any(kw in t for kw in ["follow me", "come with me", "come along"]):
        return {"intent": "follow_me"}

    # describe surroundings
    if any(kw in t for kw in ["describe", "what do you see", "look around", "surroundings"]):
        return {"intent": "describe_surroundings"}

    # movement
    if any(word in t for word in ["move", "go", "drive", "turn", "forward", "backward", "back", "left", "right"]):
        actions = parse_movement_sequence(t)
        if actions:
            return {"intent": "movement_sequence", "actions": actions}

    return {"intent": "unknown"}


def parse_movement_sequence(text: str) -> List[Dict]:
    # Split on "and", "then", commas
    segments = re.split(r"\b(?:and|then)\b|,", text)
    actions: List[Dict] = []

    for raw_seg in segments:
        seg = raw_seg.strip()
        if not seg:
            continue

        if "stop" in seg:
            actions.append({"type": "stop"})
            continue

        if any(word in seg for word in ["move", "go", "drive", "forward", "backward", "back"]):
            direction = "forward"
            if "back" in seg:
                direction = "backward"

            # extract distance
            distance_m = 0.5  # default
            match = re.search(r"(\d+(?:\.\d+)?)\s*(?:m|meter|meters)?", seg)
            if match:
                distance_m = float(match.group(1))

            actions.append({
                "type": "move",
                "direction": direction,
                "distance_m": distance_m,
                "speed": 0.2,
            })
            continue

        # turn
        if "turn" in seg or "left" in seg or "right" in seg:
            direction = None
            if "left" in seg:
                direction = "left"
            elif "right" in seg:
                direction = "right"

            if direction is None:
                continue

            # extract degrees
            degrees = 90.0  # default
            match = re.search(r"(\d+(?:\.\d+)?)\s*(?:deg|degree|degrees)?", seg)
            if match:
                degrees = float(match.group(1))

            actions.append({
                "type": "turn",
                "direction": direction,
                "degrees": degrees,
            })
            continue

    return actions
