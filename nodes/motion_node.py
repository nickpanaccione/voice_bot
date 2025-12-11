#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import json

class MotionNode(Node):
    def __init__(self):
        super().__init__("motion_node")
        self.sub = self.create_subscription(String, "nlu/intent", self.cb, 10)
        self.pub = self.create_publisher(Twist, "cmd_vel", 10)

        self.max_lin = 0.22
        self.max_ang = 2.84

    def clamp(self, val, lo, hi):
        return max(lo, min(hi, val))

    def cb(self, msg: String):
        try:
            intent = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f"Bad intent JSON: {e}")
            return

        action = intent.get("action", "stop")
        value = float(intent.get("value", 0.2))

        twist = Twist()

        if action == "forward":
            twist.linear.x = self.clamp(value, -self.max_lin, self.max_lin)
        elif action == "backward":
            twist.linear.x = -self.clamp(value, -self.max_lin, self.max_lin)
        elif action == "left":
            twist.angular.z = self.clamp(value, -self.max_ang, self.max_ang)
        elif action == "right":
            twist.angular.z = -self.clamp(value, -self.max_ang, self.max_ang)
        elif action == "stop":
            twist.linear.x = 0.0
            twist.angular.z = 0.0
        else:
            self.get_logger().warn(f"Unknown action: {action}")
            return

        self.pub.publish(twist)
        self.get_logger().info(f"Executed action={action}, value={value}")

def main(args=None):
    rclpy.init(args=args)
    node = MotionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()