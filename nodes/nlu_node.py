#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import requests
import json
import os

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")

class NLUGemmaOllamaNode(Node):
    def __init__(self):
        super().__init__("nlu_node")
        self.sub = self.create_subscription(String, "asr/text", self.cb, 10)
        self.pub = self.create_publisher(String, "nlu/intent", 10)
        self.get_logger().info(f"Using Ollama model: {OLLAMA_MODEL} at {OLLAMA_URL}")

    def build_prompt(self, text: str) -> str:
        return (
            "You are an intent parser for robot motion commands.\n"
            "Respond ONLY with a single line of valid JSON.\n"
            "JSON keys: action, value, units.\n"
            "Allowed actions: forward, backward, left, right, stop.\n"
            "If no numeric value is given, use 0.2 and units 'm'.\n"
            "Examples:\n"
            'Input: "forward 0.5 meters"\n'
            'Output: {"action": "forward", "value": 0.5, "units": "m"}\n'
            'Input: "spin left"\n'
            'Output: {"action": "left", "value": 0.3, "units": "rad/s"}\n'
            f'Input: "{text}"\n'
            "Output: "
        )

    def call_ollama(self, prompt: str) -> str:
        url = f"{OLLAMA_URL}/api/generate"
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
        }
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip()

    def extract_json(self, text: str) -> str:
        # try to grab the JSON object from output
        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last != -1 and last > first:
            return text[first:last+1]
        return text

    def cb(self, msg: String):
        text = msg.data
        self.get_logger().info(f"NLU received: {text}")

        prompt = self.build_prompt(text)
        try:
            raw = self.call_ollama(prompt)
        except Exception as e:
            self.get_logger().error(f"Ollama request failed: {e}")
            return

        self.get_logger().info(f"Raw Ollama output: {raw}")
        json_str = self.extract_json(raw)
        self.get_logger().info(f"Extracted JSON: {json_str}")

        try:
            intent = json.loads(json_str)
        except Exception as e:
            self.get_logger().error(f"Failed to parse JSON: {e}")
            return

        out_msg = String()
        out_msg.data = json.dumps(intent)
        self.pub.publish(out_msg)
        self.get_logger().info(f"Published intent: {out_msg.data}")

def main(args=None):
    rclpy.init(args=args)
    node = NLUGemmaOllamaNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()