#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import math
from collections import deque
import time

from sensor_msgs.msg import CompressedImage, PointCloud2, Image, Imu
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, String
from geometry_msgs.msg import Vector3, Quaternion

# Custom message import (assuming you have these defined)
# from custom_msgs.msg import ObstacleArray, Obstacle

# Additional imports for processing
import cv2
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("perception")
        
        # Initialize sensor data storage for fusion
        self.vehicle_state = {
            'position': [0.0, 0.0, 0.0],
            'orientation': [0.0, 0.0, 0.0, 1.0],  # quaternion [x, y, z, w]
            'velocity': [0.0, 0.0, 0.0],
            'angular_velocity': [0.0, 0.0, 0.0],
            'acceleration': [0.0, 0.0, 0.0],
            'last_update_time': time.time()
        }
        
        # Sensor fusion parameters
        self.imu_data_buffer = deque(maxlen=10)  # Store recent IMU readings for smoothing
        self.confidence_thresholds = {
            'lane_detection': 0.7,
            'obstacle_detection': 0.8,
            'traffic_light': 0.6
        }
        
        # Detection states for sensor fusion
        self.lane_detected = False
        self.lane_confidence = 0.0
        self.detected_obstacles = []
        self.traffic_light_state = "unknown"
        
        # Subscriptions to sensor topics
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
            self.depth_callback,
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
        
        # Publishers for perception outputs
        # Note: Uncomment these when you have custom_msgs package
        # self.obstacle_publisher = self.create_publisher(
        #     ObstacleArray,
        #     "/perception/obstacles",
        #     10
        # )
        
        self.lane_publisher = self.create_publisher(
            Bool,
            "/perception/lane_detected",
            10
        )
        
        self.traffic_light_publisher = self.create_publisher(
            String,
            "/perception/traffic_light",
            10
        )
        
        # Initialize CvBridge for image conversions
        self.bridge = CvBridge()
        
        # Create a timer for periodic sensor fusion and publishing
        self.create_timer(0.1, self.sensor_fusion_callback)  # 10Hz fusion rate
        
        self.get_logger().info("Enhanced Perception Node initialized with sensor fusion capabilities")

    def rgb_front_callback(self, msg):
        """
        Enhanced RGB image processing with lane detection and traffic light recognition.
        This callback now includes confidence scoring and multiple detection algorithms.
        """
        try:
            # Convert compressed image to OpenCV format
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if cv_image is None:
                self.get_logger().error("Failed to decode compressed image.")
                return
            
            # Perform lane detection with confidence scoring
            lane_result = self._detect_lanes_advanced(cv_image)
            self.lane_detected = lane_result['detected']
            self.lane_confidence = lane_result['confidence']
            
            # Perform traffic light detection
            traffic_light_result = self._detect_traffic_lights(cv_image)
            self.traffic_light_state = traffic_light_result['state']
            
            # Log results with confidence levels
            if self.lane_detected:
                self.get_logger().info(
                    f"Lane detected with confidence: {self.lane_confidence:.2f}, "
                    f"steering error: {lane_result['steering_error']} pixels"
                )
            
            if self.traffic_light_state != "unknown":
                self.get_logger().info(f"Traffic light detected: {self.traffic_light_state}")
                
        except Exception as e:
            self.get_logger().error(f"Error in RGB processing: {e}")

    def _detect_lanes_advanced(self, cv_image):
        """
        Advanced lane detection with multiple methods and confidence scoring.
        Returns dictionary with detection results and confidence level.
        """
        try:
            # Convert to HSV for better color filtering
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            
            # Define multiple color ranges for robustness
            white_lower = np.array([0, 0, 200], dtype=np.uint8)
            white_upper = np.array([180, 30, 255], dtype=np.uint8)
            yellow_lower = np.array([15, 80, 80], dtype=np.uint8)
            yellow_upper = np.array([35, 255, 255], dtype=np.uint8)
            
            # Create masks for different lane colors
            white_mask = cv2.inRange(hsv, white_lower, white_upper)
            yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
            combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
            
            # Apply morphological operations to clean up the mask
            kernel = np.ones((3, 3), np.uint8)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return {'detected': False, 'confidence': 0.0, 'steering_error': 0}
            
            # Filter contours by area and aspect ratio for better lane detection
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Minimum area threshold
                    # Check if contour resembles a lane (elongated shape)
                    rect = cv2.minAreaRect(contour)
                    width, height = rect[1]
                    if width > 0 and height > 0:
                        aspect_ratio = max(width, height) / min(width, height)
                        if aspect_ratio > 2:  # Lane should be elongated
                            valid_contours.append(contour)
            
            if not valid_contours:
                return {'detected': False, 'confidence': 0.0, 'steering_error': 0}
            
            # Find the largest valid contour
            largest_contour = max(valid_contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            
            if M["m00"] == 0:
                return {'detected': False, 'confidence': 0.0, 'steering_error': 0}
            
            # Calculate centroid and steering error
            cx = int(M["m10"] / M["m00"])
            image_center = cv_image.shape[1] // 2
            steering_error = cx - image_center
            
            # Calculate confidence based on contour properties
            contour_area = cv2.contourArea(largest_contour)
            max_possible_area = cv_image.shape[0] * cv_image.shape[1] * 0.3  # 30% of image
            area_confidence = min(contour_area / max_possible_area, 1.0)
            
            # Confidence also based on how centered the detection is
            center_confidence = 1.0 - abs(steering_error) / (cv_image.shape[1] / 2)
            
            # Combined confidence score
            overall_confidence = (area_confidence + center_confidence) / 2
            
            return {
                'detected': overall_confidence > self.confidence_thresholds['lane_detection'],
                'confidence': overall_confidence,
                'steering_error': steering_error
            }
            
        except Exception as e:
            self.get_logger().error(f"Error in advanced lane detection: {e}")
            return {'detected': False, 'confidence': 0.0, 'steering_error': 0}

    def _detect_traffic_lights(self, cv_image):
        """
        Basic traffic light detection using color filtering.
        This is a simplified implementation - in practice, you'd use more sophisticated methods.
        """
        try:
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            
            # Define color ranges for traffic lights
            red_lower = np.array([0, 120, 120], dtype=np.uint8)
            red_upper = np.array([10, 255, 255], dtype=np.uint8)
            red_lower2 = np.array([170, 120, 120], dtype=np.uint8)
            red_upper2 = np.array([180, 255, 255], dtype=np.uint8)
            
            yellow_lower = np.array([15, 120, 120], dtype=np.uint8)
            yellow_upper = np.array([35, 255, 255], dtype=np.uint8)
            
            green_lower = np.array([40, 120, 120], dtype=np.uint8)
            green_upper = np.array([80, 255, 255], dtype=np.uint8)
            
            # Create masks
            red_mask1 = cv2.inRange(hsv, red_lower, red_upper)
            red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
            green_mask = cv2.inRange(hsv, green_lower, green_upper)
            
            # Count pixels for each color (simple detection method)
            red_pixels = cv2.countNonZero(red_mask)
            yellow_pixels = cv2.countNonZero(yellow_mask)
            green_pixels = cv2.countNonZero(green_mask)
            
            # Determine traffic light state based on dominant color
            threshold = 100  # Minimum pixels to consider detection valid
            
            if red_pixels > threshold and red_pixels > yellow_pixels and red_pixels > green_pixels:
                return {'state': 'red'}
            elif yellow_pixels > threshold and yellow_pixels > green_pixels:
                return {'state': 'yellow'}
            elif green_pixels > threshold:
                return {'state': 'green'}
            else:
                return {'state': 'unknown'}
                
        except Exception as e:
            self.get_logger().error(f"Error in traffic light detection: {e}")
            return {'state': 'unknown'}

    def pointcloud_callback(self, msg):
        """
        Enhanced point cloud processing with obstacle classification and tracking.
        """
        try:
            # Convert point cloud to numpy array
            points = []
            for point in pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True):
                points.append([point[0], point[1], point[2], point[3]])
            
            points = np.array(points)
            
            if len(points) == 0:
                self.detected_obstacles = []
                return
            
            # Enhanced obstacle detection with clustering
            obstacles = self._detect_obstacles_clustered(points)
            
            # Fuse with IMU data for motion compensation
            compensated_obstacles = self._compensate_for_vehicle_motion(obstacles)
            
            self.detected_obstacles = compensated_obstacles
            
            # Log significant obstacles
            for obstacle in self.detected_obstacles:
                if obstacle['distance'] < 10.0:  # Only log close obstacles
                    self.get_logger().info(
                        f"Obstacle detected: distance={obstacle['distance']:.2f}m, "
                        f"position=({obstacle['x']:.2f}, {obstacle['y']:.2f}), "
                        f"confidence={obstacle['confidence']:.2f}"
                    )
                    
        except Exception as e:
            self.get_logger().error(f"Error processing point cloud: {e}")

    def _detect_obstacles_clustered(self, points):
        """
        Advanced obstacle detection using clustering algorithms.
        """
        try:
            # Filter points by height to remove ground and overhead objects
            filtered_points = points[(points[:, 2] > -1.5) & (points[:, 2] < 2.0)]
            
            if len(filtered_points) == 0:
                return []
            
            # Simple clustering based on distance (DBSCAN would be better in practice)
            obstacles = []
            processed = np.zeros(len(filtered_points), dtype=bool)
            
            for i, point in enumerate(filtered_points):
                if processed[i]:
                    continue
                
                # Find nearby points (simple clustering)
                distances = np.sqrt(np.sum((filtered_points - point) ** 2, axis=1))
                cluster_mask = distances < 1.0  # 1 meter clustering radius
                cluster_points = filtered_points[cluster_mask]
                processed[cluster_mask] = True
                
                if len(cluster_points) > 5:  # Minimum points for valid obstacle
                    # Calculate obstacle properties
                    centroid = np.mean(cluster_points, axis=0)
                    distance = np.sqrt(centroid[0]**2 + centroid[1]**2)
                    
                    # Calculate confidence based on point density and distance
                    confidence = min(len(cluster_points) / 50.0, 1.0)  # More points = higher confidence
                    confidence *= max(0.1, 1.0 - distance / 50.0)  # Closer = higher confidence
                    
                    obstacle = {
                        'x': centroid[0],
                        'y': centroid[1],
                        'z': centroid[2],
                        'distance': distance,
                        'point_count': len(cluster_points),
                        'confidence': confidence
                    }
                    obstacles.append(obstacle)
            
            return obstacles
            
        except Exception as e:
            self.get_logger().error(f"Error in clustered obstacle detection: {e}")
            return []

    def _compensate_for_vehicle_motion(self, obstacles):
        """
        Compensate obstacle positions for vehicle motion using IMU data.
        """
        try:
            if not self.imu_data_buffer:
                return obstacles
            
            # Get latest IMU data
            latest_imu = self.imu_data_buffer[-1]
            dt = time.time() - latest_imu['timestamp']
            
            # Simple motion compensation (in practice, use Kalman filtering)
            compensated_obstacles = []
            for obstacle in obstacles:
                # Compensate for vehicle velocity
                compensated_x = obstacle['x'] - self.vehicle_state['velocity'][0] * dt
                compensated_y = obstacle['y'] - self.vehicle_state['velocity'][1] * dt
                
                compensated_obstacle = obstacle.copy()
                compensated_obstacle['x'] = compensated_x
                compensated_obstacle['y'] = compensated_y
                compensated_obstacle['distance'] = np.sqrt(compensated_x**2 + compensated_y**2)
                
                compensated_obstacles.append(compensated_obstacle)
            
            return compensated_obstacles
            
        except Exception as e:
            self.get_logger().error(f"Error in motion compensation: {e}")
            return obstacles

    def depth_callback(self, msg):
        """
        Enhanced depth image processing with better obstacle detection.
        """
        try:
            # Convert ROS Image to OpenCV format
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
            
            # Get image dimensions
            height, width = depth_image.shape
            
            # Define multiple regions of interest for comprehensive scanning
            regions = [
                {'name': 'center', 'x1': width//3, 'x2': 2*width//3, 'y1': height//3, 'y2': 2*height//3},
                {'name': 'left', 'x1': 0, 'x2': width//3, 'y1': height//4, 'y2': 3*height//4},
                {'name': 'right', 'x1': 2*width//3, 'x2': width, 'y1': height//4, 'y2': 3*height//4}
            ]
            
            closest_distance = float('inf')
            closest_region = None
            
            for region in regions:
                roi = depth_image[region['y1']:region['y2'], region['x1']:region['x2']]
                valid_roi = roi[roi > 0]  # Remove invalid depth values
                
                if valid_roi.size > 0:
                    min_distance = np.min(valid_roi)
                    if min_distance < closest_distance:
                        closest_distance = min_distance
                        closest_region = region['name']
            
            # React based on closest obstacle
            if closest_distance < float('inf'):
                if closest_distance < 3.0:
                    self.get_logger().warn(
                        f"CRITICAL: Obstacle at {closest_distance:.2f}m in {closest_region} region!"
                    )
                elif closest_distance < 8.0:
                    self.get_logger().info(
                        f"Obstacle detected at {closest_distance:.2f}m in {closest_region} region"
                    )
                    
        except Exception as e:
            self.get_logger().error(f"Error processing depth image: {e}")

    def imu_callback(self, msg):
        """
        Robust IMU data processing with filtering and integration.
        """
        try:
            current_time = time.time()
            
            # Extract IMU data
            imu_data = {
                'timestamp': current_time,
                'linear_acceleration': [
                    msg.linear_acceleration.x,
                    msg.linear_acceleration.y,
                    msg.linear_acceleration.z
                ],
                'angular_velocity': [
                    msg.angular_velocity.x,
                    msg.angular_velocity.y,
                    msg.angular_velocity.z
                ],
                'orientation': [
                    msg.orientation.x,
                    msg.orientation.y,
                    msg.orientation.z,
                    msg.orientation.w
                ]
            }
            
            # Add to buffer for smoothing
            self.imu_data_buffer.append(imu_data)
            
            # Update vehicle state with smoothed IMU data
            self._update_vehicle_state_from_imu()
            
            # Check for significant events
            accel_magnitude = np.linalg.norm(imu_data['linear_acceleration'])
            if accel_magnitude > 15.0:  # High acceleration/deceleration
                self.get_logger().warn(f"High acceleration detected: {accel_magnitude:.2f} m/s²")
            
            angular_vel_magnitude = np.linalg.norm(imu_data['angular_velocity'])
            if angular_vel_magnitude > 2.0:  # Sharp turn
                self.get_logger().info(f"Sharp turn detected: {angular_vel_magnitude:.2f} rad/s")
                
        except Exception as e:
            self.get_logger().error(f"Error processing IMU data: {e}")

    def _update_vehicle_state_from_imu(self):
        """
        Update vehicle state using smoothed IMU data.
        """
        try:
            if len(self.imu_data_buffer) < 2:
                return
            
            # Get recent IMU data for smoothing
            recent_data = list(self.imu_data_buffer)[-5:]  # Last 5 readings
            
            # Smooth acceleration data
            smooth_accel = np.mean([data['linear_acceleration'] for data in recent_data], axis=0)
            self.vehicle_state['acceleration'] = smooth_accel.tolist()
            
            # Smooth angular velocity
            smooth_angular_vel = np.mean([data['angular_velocity'] for data in recent_data], axis=0)
            self.vehicle_state['angular_velocity'] = smooth_angular_vel.tolist()
            
            # Use latest orientation (quaternion)
            latest_orientation = recent_data[-1]['orientation']
            self.vehicle_state['orientation'] = latest_orientation
            
            # Convert quaternion to Euler angles for easier interpretation
            r = Rotation.from_quat(latest_orientation)
            euler_angles = r.as_euler('xyz', degrees=True)
            
            # Log significant orientation changes
            if abs(euler_angles[2]) > 30:  # Significant yaw change
                self.get_logger().info(f"Vehicle heading: {euler_angles[2]:.1f}° from north")
                
        except Exception as e:
            self.get_logger().error(f"Error updating vehicle state from IMU: {e}")

    def odometry_callback(self, msg):
        """
        Robust odometry processing with velocity estimation and position tracking.
        """
        try:
            current_time = time.time()
            dt = current_time - self.vehicle_state['last_update_time']
            
            # Extract position data
            new_position = [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z
            ]
            
            # Calculate velocity if we have previous position
            if dt > 0 and self.vehicle_state['last_update_time'] > 0:
                velocity = [
                    (new_position[0] - self.vehicle_state['position'][0]) / dt,
                    (new_position[1] - self.vehicle_state['position'][1]) / dt,
                    (new_position[2] - self.vehicle_state['position'][2]) / dt
                ]
                self.vehicle_state['velocity'] = velocity
                
                # Calculate speed
                speed = np.linalg.norm(velocity)
                if speed > 0.1:  # Only log if moving
                    self.get_logger().debug(f"Vehicle speed: {speed:.2f} m/s ({speed * 3.6:.1f} km/h)")
            
            # Update position and orientation
            self.vehicle_state['position'] = new_position
            self.vehicle_state['orientation'] = [
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w
            ]
            self.vehicle_state['last_update_time'] = current_time
            
            # Extract twist (velocity) data if available
            if hasattr(msg, 'twist') and msg.twist is not None:
                linear_vel = [
                    msg.twist.twist.linear.x,
                    msg.twist.twist.linear.y,
                    msg.twist.twist.linear.z
                ]
                angular_vel = [
                    msg.twist.twist.angular.x,
                    msg.twist.twist.angular.y,
                    msg.twist.twist.angular.z
                ]
                
                # Update vehicle state with odometry velocity (more accurate than calculated)
                self.vehicle_state['velocity'] = linear_vel
                # Angular velocity from odometry can complement IMU data
                
        except Exception as e:
            self.get_logger().error(f"Error processing odometry data: {e}")

    def sensor_fusion_callback(self):
        """
        Main sensor fusion loop that combines all sensor data and publishes results.
        This runs at 10Hz to provide consistent output to the planning node.
        """
        try:
            # Publish lane detection status
            lane_msg = Bool()
            lane_msg.data = self.lane_detected and self.lane_confidence > self.confidence_thresholds['lane_detection']
            self.lane_publisher.publish(lane_msg)
            
            # Publish traffic light status
            traffic_msg = String()
            traffic_msg.data = self.traffic_light_state
            self.traffic_light_publisher.publish(traffic_msg)
            
            # Publish obstacle array (uncomment when custom_msgs is available)
            # obstacle_array_msg = ObstacleArray()
            # obstacle_array_msg.header.stamp = self.get_clock().now().to_msg()
            # obstacle_array_msg.header.frame_id = "base_link"
            # 
            # for obstacle in self.detected_obstacles:
            #     if obstacle['confidence'] > self.confidence_thresholds['obstacle_detection']:
            #         obstacle_msg = Obstacle()
            #         obstacle_msg.position.x = obstacle['x']
            #         obstacle_msg.position.y = obstacle['y']
            #         obstacle_msg.position.z = obstacle['z']
            #         obstacle_msg.distance = obstacle['distance']
            #         obstacle_msg.confidence = obstacle['confidence']
            #         obstacle_array_msg.obstacles.append(obstacle_msg)
            # 
            # self.obstacle_publisher.publish(obstacle_array_msg)
            
            # Log fusion summary periodically (every 5 seconds)
            if int(time.time()) % 5 == 0:
                self.get_logger().info(
                    f"Sensor Fusion Summary - Lane: {self.lane_detected} "
                    f"({self.lane_confidence:.2f}), Obstacles: {len(self.detected_obstacles)}, "
                    f"Traffic Light: {self.traffic_light_state}, "
                    f"Speed: {np.linalg.norm(self.vehicle_state['velocity']):.1f} m/s"
                )
                
        except Exception as e:
            self.get_logger().error(f"Error in sensor fusion: {e}")

    def publish_brake_command(self, intensity):
        """Enhanced brake command with safety checks."""
        # Add safety logic here
        if intensity > 1.0:
            intensity = 1.0
        elif intensity < 0.0:
            intensity = 0.0
            
        self.get_logger().info(f"Publishing brake command: {intensity:.2f}")
        # In a real implementation, publish to appropriate control topic

    def publish_throttle_command(self, intensity):
        """Enhanced throttle command with safety checks."""
        # Add safety logic here
        if intensity > 1.0:
            intensity = 1.0
        elif intensity < 0.0:
            intensity = 0.0
            
        self.get_logger().info(f"Publishing throttle command: {intensity:.2f}")
        # In a real implementation, publish to appropriate control topic

    def publish_steering_command(self, angle):
        """Enhanced steering command with safety checks."""
        # Limit steering angle for safety
        max_steering = 0.5  # radians
        if angle > max_steering:
            angle = max_steering
        elif angle < -max_steering:
            angle = -max_steering
            
        self.get_logger().info(f"Publishing steering command: {angle:.3f} rad")
        # In a real implementation, publish to appropriate control topic


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Perception node shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()