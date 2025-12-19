#!/usr/bin/env python3
import json
import requests

from robot import config

class LLMIntentError(Exception):
    pass

class LLMIntentParser:
    def __init__(self):
        self.url = config.OLLAMA_URL
        self.model = config.OLLAMA_MODEL
        self.timeout = config.OLLAMA_TIMEOUT

    def build_prompt(self, text: str) -> str:
        """Build a concise prompt with examples for small models."""
        return (
            "Parse this robot command into JSON. ONLY include actions explicitly mentioned.\n"
            "\n"
            "Examples:\n"
            'Input: "move forward 2 meters" -> {"intent":"movement_sequence","actions":[{"type":"move","direction":"forward","distance_m":2.0}]}\n'
            'Input: "turn right" -> {"intent":"movement_sequence","actions":[{"type":"turn","direction":"right","degrees":90}]}\n'
            'Input: "go back 1 meter then turn left" -> {"intent":"movement_sequence","actions":[{"type":"move","direction":"backward","distance_m":1.0},{"type":"turn","direction":"left","degrees":90}]}\n'
            'Input: "move forward 1 meter then turn right 90 degrees then move forward 1 meter" -> {"intent":"movement_sequence","actions":[{"type":"move","direction":"forward","distance_m":1.0},{"type":"turn","direction":"right","degrees":90},{"type":"move","direction":"forward","distance_m":1.0}]}\n'
            'Input: "follow me" or "come with me" -> {"intent":"follow_me"}\n'
            'Input: "describe" or "what do you see" -> {"intent":"describe_surroundings"}\n'
            'Input: "stop" -> {"intent":"stop"}\n'
            'Input: "shut down" -> {"intent":"shutdown"}\n'
            'Input: "help" or "what can you do" -> {"intent":"help"}\n'
            "\n"
            "Rules:\n"
            "- Output ONLY valid JSON, nothing else\n"
            "- ONLY include actions the user explicitly said\n"
            "- Convert word numbers: one=1, two=2, three=3, four=4, five=5, six=6, seven=7, eight=8, nine=9, ten=10\n"
            "- Default distance is 0.5 meters, default turn is 90 degrees\n"
            "- For unclear commands use: {\"intent\":\"unknown\"}\n"
            f'\nInput: "{text}"\nOutput: '
        )

    def call_ollama(self, prompt: str) -> str:
        url = f"{self.url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        resp = requests.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip()

    @staticmethod
    def _extract_json(text: str) -> str:
        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last != -1 and last > first:
            return text[first:last + 1]
        return text

    def parse(self, text: str) -> dict:
        prompt = self.build_prompt(text)
        raw = self.call_ollama(prompt)
        json_str = self._extract_json(raw)

        try:
            intent = json.loads(json_str)
        except Exception as e:
            raise LLMIntentError(f"Failed to parse JSON from LLM: {e}; raw={raw!r}")

        if "intent" not in intent:
            intent["intent"] = "unknown"

        # normalize movement_sequence
        if intent["intent"] == "movement_sequence":
            actions = intent.get("actions") or []
            normalized_actions = []
            for a in actions:
                atype = a.get("type")
                if atype == "move":
                    direction = a.get("direction", "forward")
                    dist = float(a.get("distance_m", 0.5))
                    speed = float(a.get("speed", 0.2))
                    normalized_actions.append({
                        "type": "move",
                        "direction": direction,
                        "distance_m": dist,
                        "speed": speed,
                    })
                elif atype == "turn":
                    direction = a.get("direction", "right")
                    degrees = float(a.get("degrees", 90.0))
                    normalized_actions.append({
                        "type": "turn",
                        "direction": direction,
                        "degrees": degrees,
                    })
                elif atype == "stop":
                    normalized_actions.append({"type": "stop"})
            intent["actions"] = normalized_actions

        return intent
