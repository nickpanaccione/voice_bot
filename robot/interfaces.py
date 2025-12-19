#!/usr/bin/env python3
import math
import subprocess
import time
import threading
from pathlib import Path

from robot import config
from .vision import describe_scene_with_gemini

import rclpy
from geometry_msgs.msg import Twist

BASE_DIR = Path(__file__).resolve().parent.parent
AUDIO_DIR = BASE_DIR / "data" / "audio"


def _play_sound(path: Path):
    # play sound file
    if not path.exists():
        print(f"[SOUND WARN] Missing: {path}")
        return

    player = config.AUDIO_PLAYER
    try:
        subprocess.Popen(
            [player, str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        try:
            subprocess.Popen(
                ["aplay", str(path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            print(f"[SOUND WARN] No audio player found")


class Create3RobotInterface:
    def __init__(self, node_name: str = "voice_robot"):
        # sound files
        self.wake_sound = AUDIO_DIR / "wake.wav"
        self.sleep_sound = AUDIO_DIR / "sleep.wav"
        self.thought_sound = AUDIO_DIR / "thought.wav"
        self.thought_complete_sound = AUDIO_DIR / "thoughtcomplete.wav"

        # ROS2 setup
        print("[ROS] Initializing rclpy...")
        rclpy.init(args=None)
        self._node = rclpy.create_node(node_name)
        self._pub_cmd_vel = self._node.create_publisher(Twist, "/cmd_vel", 10)

        # motion parameters
        self.default_lin_speed = config.DEFAULT_LINEAR_SPEED
        self.default_ang_speed = config.DEFAULT_ANGULAR_SPEED
        self.move_compensation = config.MOVE_COMPENSATION
        self.turn_compensation = config.TURN_COMPENSATION

        # interrupt flag
        self._interrupt_movement = threading.Event()
        self._is_moving = threading.Event()

        # human followe
        self._follower = None

        print("[ROS] Create3 interface ready, publishing to /cmd_vel")

    # tts 
    def speak(self, text: str):
        print(f"[TTS] {text}")
        try:
            subprocess.run(
                ["espeak-ng", "-v", "en-us", "-s", "150", text],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            try:
                subprocess.run(
                    ["espeak", text],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except FileNotFoundError:
                print("[TTS WARN] No TTS engine found (install espeak-ng)")

    # sfx 
    def play_ack_sound(self):
        _play_sound(self.wake_sound)

    def play_sleep_sound(self):
        _play_sound(self.sleep_sound)

    def play_processing_start(self):
        _play_sound(self.thought_sound)

    def play_processing_stop(self):
        _play_sound(self.thought_complete_sound)

    # vision
    def describe_scene(self) -> str:
        return describe_scene_with_gemini()

    # following
    def start_following(self):
        from robot.follow import HumanFollower, FollowConfig
        
        if self._follower and self._follower.is_running():
            print("[FOLLOW] Already following")
            return

        follow_config = FollowConfig(
            camera_device=config.CAMERA_DEVICE,
            max_linear_speed=config.DEFAULT_LINEAR_SPEED,
            max_angular_speed=config.DEFAULT_ANGULAR_SPEED,
        )
        
        self._follower = HumanFollower(self._pub_cmd_vel, follow_config)
        self._follower.start()
        self._is_moving.set()

    def stop_following(self):
        if self._follower:
            self._follower.stop()
            self._follower = None
        self._is_moving.clear()

    def is_following(self) -> bool:
        return self._follower is not None and self._follower.is_running()

    # movement 
    def is_moving(self) -> bool:
        return self._is_moving.is_set() or self.is_following()

    def interrupt(self):
        if self.is_following():
            self.stop_following()
        elif self._is_moving.is_set():
            print("[MOVE] Interrupt requested!")
            self._interrupt_movement.set()

    def _publish_for_duration(self, twist: Twist, duration: float, rate_hz: float = 10.0):
        period = 1.0 / rate_hz
        end_time = time.time() + max(duration, 0.0)

        while time.time() < end_time:
            if self._interrupt_movement.is_set():
                print("[MOVE] Movement interrupted!")
                break
            
            self._pub_cmd_vel.publish(twist)
            rclpy.spin_once(self._node, timeout_sec=0.0)
            time.sleep(period)

    def stop(self):
        if self.is_following():
            self.stop_following()
        
        self._interrupt_movement.set()
        twist = Twist()
        self._pub_cmd_vel.publish(twist)
        rclpy.spin_once(self._node, timeout_sec=0.0)
        self._is_moving.clear()

    def execute_movement_sequence(self, actions):
        print("[MOVE] Executing sequence...")
        self._interrupt_movement.clear()
        self._is_moving.set()

        try:
            for i, action in enumerate(actions, start=1):
                if self._interrupt_movement.is_set():
                    print("[MOVE] Sequence interrupted!")
                    break

                atype = action.get("type")
                print(f"  Step {i}: {action}")

                if atype == "move":
                    direction = action.get("direction", "forward")
                    distance = float(action.get("distance_m") or 0.5)
                    speed = float(action.get("speed") or self.default_lin_speed)

                    if distance <= 0 or speed <= 0:
                        continue

                    duration = (distance / speed) * self.move_compensation

                    twist = Twist()
                    twist.linear.x = speed if direction == "forward" else -speed

                    print(f"    -> {direction} {distance:.2f}m at {speed:.2f}m/s for {duration:.1f}s")
                    self._publish_for_duration(twist, duration)
                    self._stop_motors()

                elif atype == "turn":
                    direction = action.get("direction", "right")
                    degrees = float(action.get("degrees") or 90.0)
                    speed = self.default_ang_speed

                    if degrees == 0:
                        continue

                    radians = math.radians(degrees) * self.turn_compensation
                    duration = radians / speed

                    twist = Twist()
                    twist.angular.z = speed if direction == "left" else -speed

                    print(f"    -> {direction} {degrees:.0f}° at {speed:.2f}rad/s for {duration:.1f}s")
                    self._publish_for_duration(twist, duration)
                    self._stop_motors()

                elif atype == "stop":
                    print("    -> Stop")
                    self._stop_motors()

                time.sleep(0.2)

        finally:
            self._stop_motors()
            self._is_moving.clear()
            self._interrupt_movement.clear()

        print("[MOVE] Sequence complete")

    def _stop_motors(self):
        twist = Twist()
        self._pub_cmd_vel.publish(twist)
        rclpy.spin_once(self._node, timeout_sec=0.0)

    # shutdown 
    def shutdown(self):
        """Clean shutdown."""
        print("[ROS] Shutting down...")
        try:
            self.stop()
        except Exception:
            pass
        try:
            self._node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass
