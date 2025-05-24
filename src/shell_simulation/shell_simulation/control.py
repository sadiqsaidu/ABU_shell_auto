#!/usr/bin/env python3
import rclpy
import math
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32

class VehicleController(Node):
    def __init__(self):
        super().__init__('pid_vehicle_controller')
        
        # Subscribers
        self.waypoint_sub = self.create_subscription(
            PoseStamped,
            '/planning/next_waypoint',
            self.waypoint_callback,
            10)
            
        self.odom_sub = self.create_subscription(
            Odometry,
            '/carla/ego_vehicle/odometry',
            self.odom_callback,
            10)
            
        self.speed_sub = self.create_subscription(
            Float32,
            '/carla/ego_vehicle/speedometer',
            self.speed_callback,
            10)

        # Publishers
        self.throttle_pub = self.create_publisher(Float32, '/throttle_command', 10)
        self.brake_pub = self.create_publisher(Float32, '/brake_command', 10)
        self.steer_pub = self.create_publisher(Float32, '/steering_command', 10)
        self.gear_pub = self.create_publisher(Float32, '/gear_command', 10)

        # PID Control Parameters
        # Steering Control
        self.steering_kp = 0.8
        self.steering_ki = 0.01
        self.steering_kd = 0.2
        self.steering_integral = 0.0
        self.prev_steering_error = 0.0
        self.steering_windup_guard = 1.0

        # Speed Control
        self.speed_kp = 0.4
        self.speed_ki = 0.05
        self.speed_kd = 0.1
        self.speed_integral = 0.0
        self.prev_speed_error = 0.0
        self.speed_windup_guard = 2.0

        # Operational Parameters
        self.dt = 0.05  # 20Hz control loop
        self.max_throttle = 0.8
        self.max_brake = 0.5
        self.current_speed = 0.0
        self.current_pose = None
        self.target_waypoint = None

    def waypoint_callback(self, msg):
        self.target_waypoint = msg.pose
        self.get_logger().info(f"New target waypoint: {self.target_waypoint.position}")
        # Reset integrators when new waypoint received
        self.steering_integral = 0.0
        self.speed_integral = 0.0

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose
        self.get_logger().debug("Updated vehicle odometry")

    def speed_callback(self, msg):
        self.current_speed = msg.data
        self.get_logger().debug(f"Current speed: {self.current_speed} m/s")

    def calculate_steering(self):
        if self.current_pose is None or self.target_waypoint is None:
            return 0.0

        # Calculate position error
        dx = self.target_waypoint.position.x - self.current_pose.position.x
        dy = self.target_waypoint.position.y - self.current_pose.position.y
        current_error = math.atan2(dy, dx)

        # PID components
        p_term = self.steering_kp * current_error
        
        # Integral term with anti-windup
        self.steering_integral += current_error * self.dt
        self.steering_integral = max(min(self.steering_integral, 
                                      self.steering_windup_guard),
                                  -self.steering_windup_guard)
        i_term = self.steering_ki * self.steering_integral
        
        # Derivative term
        d_error = (current_error - self.prev_steering_error) / self.dt
        d_term = self.steering_kd * d_error
        
        # Store previous error
        self.prev_steering_error = current_error

        # Combine terms and clamp
        steering = p_term + i_term + d_term
        return max(min(steering, 1.0), -1.0)

    def calculate_throttle_brake(self, target_speed):
        speed_error = target_speed - self.current_speed

        # PID components
        p_term = self.speed_kp * speed_error
        
        # Integral term with anti-windup
        self.speed_integral += speed_error * self.dt
        self.speed_integral = max(min(self.speed_integral, 
                                    self.speed_windup_guard),
                                -self.speed_windup_guard)
        i_term = self.speed_ki * self.speed_integral
        
        # Derivative term
        d_error = (speed_error - self.prev_speed_error) / self.dt
        d_term = self.speed_kd * d_error
        
        # Store previous error
        self.prev_speed_error = speed_error

        # Combine terms
        control_output = p_term + i_term + d_term

        # Split into throttle/brake
        if control_output > 0:
            throttle = min(control_output, self.max_throttle)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(abs(control_output), self.max_brake)

        return throttle, brake

    def control_loop(self):
        if self.target_waypoint is None:
            return

        # Extract target speed from waypoint's Z coordinate
        target_speed = self.target_waypoint.position.z
        
        # Calculate controls
        steering = self.calculate_steering()
        throttle, brake = self.calculate_throttle_brake(target_speed)
        
        # Publish commands
        self.steer_pub.publish(Float32(data=steering))
        self.throttle_pub.publish(Float32(data=throttle))
        self.brake_pub.publish(Float32(data=brake))
        self.gear_pub.publish(Float32(data=1.0))  # Default forward gear

def main(args=None):
    rclpy.init(args=args)
    controller = VehicleController()
    
    # Run control loop at 20Hz (0.05s interval)
    timer = controller.create_timer(controller.dt, controller.control_loop)
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()