#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from piper import PiperVoice
from pathlib import Path
import wave

OUT_WAV = Path("/data/tts_output.wav")

class TTSNode(Node):
    def __init__(self):
        super().__init__("tts_node")
        self.sub = self.create_subscription(String, "tts/say", self.cb, 10)

        self.declare_parameter("voice_path", "/models/en_US-amy-medium.onnx")
        voice_path = self.get_parameter("voice_path").get_parameter_value().string_value

        self.get_logger().info(f"Loading Piper voice from {voice_path}")
        config_path = voice_path + ".json"
        self.get_logger().info(f"Using Piper config: {config_path}")

        self.voice = PiperVoice.load(voice_path, config_path=config_path)

    def cb(self, msg: String):
        text = msg.data
        self.get_logger().info(f"TTS saying: {text}")

        # Open WAV file and let Piper write into it
        with wave.open(str(OUT_WAV), "wb") as wav_file:
            self.voice.synthesize(
                text=text,
                wav_file=wav_file,
                speaker_id=None,
                length_scale=None,
                noise_scale=None,
                noise_w=None,
                sentence_silence=0.0,
            )

        self.get_logger().info(f"TTS wrote {OUT_WAV}")

def main(args=None):
    rclpy.init(args=args)
    node = TTSNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()