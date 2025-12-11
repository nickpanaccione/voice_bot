#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import whisper
from pathlib import Path

AUDIO_PATH = Path("/data/command.wav")  # mounted from host

class ASRNode(Node):
    def __init__(self):
        super().__init__("asr_node")
        self.pub = self.create_publisher(String, "asr/text", 10)
        self.declare_parameter("model_name", "tiny")
        model_name = self.get_parameter("model_name").get_parameter_value().string_value
        self.get_logger().info(f"Loading Whisper model: {model_name}")
        self.model = whisper.load_model(model_name)

        # run once shortly after startup
        self.timer = self.create_timer(2.0, self.run_once)
        self.ran = False

    def run_once(self):
        if self.ran:
            return
        self.ran = True

        if not AUDIO_PATH.exists():
            self.get_logger().error(f"Audio file not found: {AUDIO_PATH}")
            return

        self.get_logger().info(f"Reading audio from {AUDIO_PATH}")
        # Use whisper's loader so dtype is correct (float32) and resampled to 16k
        audio = whisper.load_audio(str(AUDIO_PATH))
        result = self.model.transcribe(audio, fp16=False)
        text = result["text"].strip()
        self.get_logger().info(f"ASR text: {text}")
        msg = String()
        msg.data = text
        self.pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ASRNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()