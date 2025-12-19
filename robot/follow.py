#!/usr/bin/env python3
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("[FOLLOW] MediaPipe not available, will use OpenCV HOG fallback")


@dataclass
class PersonDetection:
    bbox: Tuple[int, int, int, int]
    center_x: int
    center_y: int
    area: int
    confidence: float
    normalized_x: float = 0.0
    normalized_area: float = 0.0


@dataclass
class FollowConfig:
    camera_device: str = "/dev/video0"
    angular_kp: float = 0.8
    max_angular_speed: float = 1.0
    angular_deadzone: float = 0.1
    max_linear_speed: float = 0.2
    search_speed: float = 0.3
    loop_rate: float = 5.0  # Hz - CPU friendly


class PersonDetector:
    def __init__(self, use_mediapipe: bool = True):
        self.use_mediapipe = use_mediapipe and MEDIAPIPE_AVAILABLE
        self.frame_width = 320
        self.frame_height = 240

        # init HOG as fallback
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        if self.use_mediapipe:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=0,
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3,
            )
            print("[FOLLOW] Using MediaPipe detector")
        else:
            print("[FOLLOW] Using HOG detector")

    def detect(self, frame: np.ndarray) -> Optional[PersonDetection]:
        self.frame_height, self.frame_width = frame.shape[:2]

        detection = None

        if self.use_mediapipe:
            detection = self._detect_mediapipe(frame)

        if detection is None:
            detection = self._detect_hog(frame)

        return detection

    def _detect_mediapipe(self, frame: np.ndarray) -> Optional[PersonDetection]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        if not results.pose_landmarks:
            return None

        landmarks = results.pose_landmarks.landmark
        x_coords = []
        y_coords = []

        for lm in landmarks:
            if lm.visibility > 0.2:
                x_coords.append(lm.x * self.frame_width)
                y_coords.append(lm.y * self.frame_height)

        if len(x_coords) < 3:
            return None

        padding = 40
        x_min = int(max(0, min(x_coords) - padding))
        x_max = int(min(self.frame_width, max(x_coords) + padding))
        y_min = int(max(0, min(y_coords) - padding))
        y_max = int(min(self.frame_height, max(y_coords) + padding))

        width = x_max - x_min
        height = y_max - y_min
        center_x = x_min + width // 2
        center_y = y_min + height // 2
        area = width * height

        avg_visibility = sum(lm.visibility for lm in landmarks) / len(landmarks)

        detection = PersonDetection(
            bbox=(x_min, y_min, width, height),
            center_x=center_x,
            center_y=center_y,
            area=area,
            confidence=avg_visibility,
        )
        detection.normalized_x = (center_x - self.frame_width / 2) / (self.frame_width / 2)
        detection.normalized_area = area / (self.frame_width * self.frame_height)

        return detection

    def _detect_hog(self, frame: np.ndarray) -> Optional[PersonDetection]:
        scale = 0.75
        small_frame = cv2.resize(frame, None, fx=scale, fy=scale)

        boxes, weights = self.hog.detectMultiScale(
            small_frame,
            winStride=(4, 4),
            padding=(8, 8),
            scale=1.02,
            hitThreshold=0,
        )

        if len(boxes) == 0:
            return None

        best_idx = 0
        best_area = 0
        for i, (x, y, w, h) in enumerate(boxes):
            area = w * h
            if area > best_area:
                best_area = area
                best_idx = i

        x, y, w, h = boxes[best_idx]
        x = int(x / scale)
        y = int(y / scale)
        w = int(w / scale)
        h = int(h / scale)

        center_x = x + w // 2
        center_y = y + h // 2
        area = w * h

        detection = PersonDetection(
            bbox=(x, y, w, h),
            center_x=center_x,
            center_y=center_y,
            area=area,
            confidence=float(weights[best_idx]) if len(weights) > 0 else 0.5,
        )
        detection.normalized_x = (center_x - self.frame_width / 2) / (self.frame_width / 2)
        detection.normalized_area = area / (self.frame_width * self.frame_height)

        return detection

    def close(self):
        # release resources
        if self.use_mediapipe and hasattr(self, 'pose'):
            self.pose.close()


class HumanFollower:
    def __init__(self, cmd_vel_publisher, config: FollowConfig = None):
        self.config = config or FollowConfig()
        self.cmd_vel_pub = cmd_vel_publisher
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        self.detector: Optional[PersonDetector] = None
        self.cap: Optional[cv2.VideoCapture] = None
        
        # Search state
        self._search_direction = 1
        self._search_time = 0.0
        self._frames_without_detection = 0

    def start(self):
        if self._running:
            print("[FOLLOW] Already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print("[FOLLOW] Started following")

    def stop(self):
        if not self._running:
            return

        print("[FOLLOW] Stopping...")
        self._stop_event.set()
        
        if self._thread:
            self._thread.join(timeout=2.0)
        
        self._running = False
        print("[FOLLOW] Stopped")

    def is_running(self) -> bool:
        return self._running

    def _init_camera(self):
        print(f"[FOLLOW] Opening camera {self.config.camera_device}...")
        
        # Try device path first
        self.cap = cv2.VideoCapture(self.config.camera_device)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")

        # Low resolution for CPU efficiency
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.cap.set(cv2.CAP_PROP_FPS, 15)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Warm up
        for _ in range(5):
            self.cap.read()

        print("[FOLLOW] Camera ready")

    def _send_velocity(self, linear: float, angular: float):
        from geometry_msgs.msg import Twist
        twist = Twist()
        twist.linear.x = float(linear)
        twist.angular.z = float(angular)
        self.cmd_vel_pub.publish(twist)

    def _compute_velocity(self, detection: Optional[PersonDetection], dt: float) -> Tuple[float, float]:
        if detection is None:
            self._frames_without_detection += 1
            
            # Search if lost for a while
            if self._frames_without_detection > 10:
                self._search_time += dt
                if self._search_time > 5.0:
                    self._search_direction *= -1
                    self._search_time = 0.0
                return (0.0, self._search_direction * self.config.search_speed)
            
            return (0.0, 0.0)

        # Person found
        self._frames_without_detection = 0
        self._search_time = 0.0

        # Angular velocity (turn toward person)
        x_error = detection.normalized_x
        if abs(x_error) < self.config.angular_deadzone:
            angular_vel = 0.0
        else:
            angular_vel = -self.config.angular_kp * x_error
            angular_vel = max(-self.config.max_angular_speed,
                            min(self.config.max_angular_speed, angular_vel))

        # Linear velocity (move toward person)
        area = detection.normalized_area
        if area > 0.4:
            linear_vel = self.config.max_linear_speed * 0.3
        elif area > 0.2:
            linear_vel = self.config.max_linear_speed * 0.5
        else:
            linear_vel = self.config.max_linear_speed

        return (linear_vel, angular_vel)

    def _run_loop(self):
        self._running = True
        
        try:
            self._init_camera()
            self.detector = PersonDetector(use_mediapipe=MEDIAPIPE_AVAILABLE)
            
            rate_period = 1.0 / self.config.loop_rate
            last_time = time.time()
            frame_skip = 0

            print("[FOLLOW] Following active - looking for person...")

            while not self._stop_event.is_set():
                loop_start = time.time()
                dt = loop_start - last_time
                last_time = loop_start

                # Read frame
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    time.sleep(0.1)
                    continue

                # Skip frames for CPU
                frame_skip += 1
                if frame_skip < 2:
                    continue
                frame_skip = 0

                # Detect person
                detection = self.detector.detect(frame)

                # Compute and send velocity
                linear, angular = self._compute_velocity(detection, dt)
                self._send_velocity(linear, angular)

                # Status logging
                if detection:
                    direction = "LEFT" if detection.normalized_x < -0.1 else "RIGHT" if detection.normalized_x > 0.1 else "CENTER"
                    print(f"[FOLLOW] Person: {direction} | lin={linear:.2f} ang={angular:.2f}")

                # Rate limit
                elapsed = time.time() - loop_start
                sleep_time = max(0.05, rate_period - elapsed)
                time.sleep(sleep_time)

        except Exception as e:
            print(f"[FOLLOW] Error: {e}")
        finally:
            # Stop robot
            self._send_velocity(0.0, 0.0)
            
            # Cleanup
            if self.cap:
                self.cap.release()
            if self.detector:
                self.detector.close()
            
            self._running = False
            print("[FOLLOW] Loop ended")
