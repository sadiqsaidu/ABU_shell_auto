#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

# Import the required message types
from sensor_msgs.msg import CompressedImage, PointCloud2, Image, Imu
from nav_msgs.msg import Odometry

# Additional imports for lane detection
import cv2
import numpy as np

class PerceptionNode(Node):
    def __init__(self):
        super().__init__("perception")

        # Subscriptions to the specified topics
        self.create_subscription(
            CompressedImage,
            "/carla/ego_vehicle/rgb_front/image/compressed",
            self.rgb_front_callback,
            10
        )
        self.create_subscription(
            PointCloud2,
            "/carla/ego_vehicle/vlp16_1",
            self.pointcloud_callback,
            10
        )
        self.create_subscription(
            Image,
            "/carla/ego_vehicle/depth_middle/image",
            self.depth_image_callback,
            10
        )
        self.create_subscription(
            Imu,
            "/carla/ego_vehicle/imu",
            self.imu_callback,
            10
        )
        self.create_subscription(
            Odometry,
            "/carla/ego_vehicle/odometry",
            self.odometry_callback,
            10
        )

    # Callback methods for each topic
    def rgb_front_callback(self, msg):
        """Process incoming images and compute steering error using advanced lane detection."""
        try:
            # Convert compressed image to OpenCV format
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if cv_image is None:
                self.get_logger().error("Failed to decode compressed image.")
                return

            # Convert to grayscale
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Perform Canny edge detection
            edges = cv2.Canny(blur, 50, 150)
            
            # Define ROI (trapezoid focusing on the road)
            height, width = edges.shape
            mask = np.zeros_like(edges)
            roi_vertices = [
                (0, height),              # Bottom-left
                (width // 2 - 50, height // 2),  # Top-left
                (width // 2 + 50, height // 2),  # Top-right
                (width, height),          # Bottom-right
            ]
            cv2.fillPoly(mask, [np.array(roi_vertices, np.int32)], 255)
            masked_edges = cv2.bitwise_and(edges, mask)
            
            # Apply Hough transform to detect lines
            lines = cv2.HoughLinesP(
                masked_edges,
                rho=1,                # Distance resolution in pixels
                theta=np.pi / 180,    # Angle resolution in radians
                threshold=50,         # Minimum votes to be considered a line
                minLineLength=100,    # Minimum length of a line
                maxLineGap=50         # Maximum gap between line segments
            )
            
            if lines is None or len(lines) == 0:
                self.get_logger().warn("No lane lines detected in the image.")
                return
            
            # Separate lines into left and right based on slope
            left_lines = []
            right_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 == 0:
                    continue  # Skip vertical lines
                slope = (y2 - y1) / (x2 - x1)
                if slope < -0.5:    # Negative slope for left lane
                    left_lines.append(line)
                elif slope > 0.5:   # Positive slope for right lane
                    right_lines.append(line)
            
            if not left_lines or not right_lines:
                self.get_logger().warn("Insufficient lane lines detected for both sides.")
                return
            
            # Average the lines for left and right lanes
            left_avg = np.mean(left_lines, axis=0)[0]
            right_avg = np.mean(right_lines, axis=0)[0]
            
            # Calculate x-position of lanes at the bottom of the image
            left_x = left_avg[0]   # Using x1 for simplicity
            right_x = right_avg[0]
            
            # Calculate lane center (midpoint between left and right)
            lane_center = (left_x + right_x) / 2
            
            # Calculate image center
            image_center = width / 2
            
            # Compute steering error
            steering_error = lane_center - image_center
            self.get_logger().info(f"Lane center offset: {steering_error:.2f} pixels")
            
        except cv2.error as e:
            self.get_logger().error(f"OpenCV error processing image: {e}")
        except Exception as e:
            self.get_logger().error(f"Unexpected error in image callback: {e}")

    def pointcloud_callback(self, msg):
        self.get_logger().info("Received PointCloud data")

    def depth_image_callback(self, msg):
        self.get_logger().info("Received Depth image")

    def imu_callback(self, msg):
        self.get_logger().info("Received IMU data")

    def odometry_callback(self, msg):
        self.get_logger().info("Received Odometry data")

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()