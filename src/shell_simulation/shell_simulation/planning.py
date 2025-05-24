#!/usr/bin/env python3
import rclpy 
from rclpy.node import Node 
import math
import heapq
import time
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix
from carla_msgs.msg import CarlaEgoVehicleStatus, CarlaTrafficLightStatusList

class PlanningNode(Node):  
    def __init__(self):
        super().__init__("planning")
        
        # Initialize waypoints and distance matrix
        self.start_point = [280.363739, -129.306351, 0.101746]
        self.waypoints = [
            [334.949799, -161.106171, 0.001736],
            [339.100037, -258.568939, 0.001679],
            [396.295319, -183.195740, 0.001678],
            [267.657074, -1.983160, 0.001678],
            [153.868896, -26.115866, 0.001678],
            [290.515564, -56.175072, 0.001677],
            [92.325722, -86.063644, 0.001677],
            [88.384346, -287.468567, 0.001728],
            [177.594101, -326.386902, 0.001677],
            [-1.646942, -197.501282, 0.001555],
            [59.701321, -1.970804, 0.001467],
            [122.100121, -55.142044, 0.001596],
            [161.030975, -129.313187, 0.001679],
            [184.758713, -199.424271, 0.001680]
        ]
        self.points = [self.start_point] + self.waypoints
        self.distance_matrix = self.precompute_distances()
        
        # Solve TSP
        self.optimal_path = self.solve_tsp()
        self.current_waypoint_idx = 0
        self.obstacles = []
        self.lane_detected = True
        self.traffic_light_status = None
        self.current_pose = None
        self.gnss_position = None

        # ROS Publishers
        self.path_pub = self.create_publisher(Path, '/planning/path', 10)
        self.next_wp_pub = self.create_publisher(PoseStamped, '/planning/next_waypoint', 10)

        # ROS Subscribers
        self.obstacles_sub = self.create_subscription(
            CarlaEgoVehicleStatus, '/perception/obstacles', self.obstacles_callback, 10)
        self.lane_sub = self.create_subscription(
            CarlaEgoVehicleStatus, '/perception/lane_detected', self.lane_callback, 10)
        self.traffic_light_sub = self.create_subscription(
            CarlaTrafficLightStatusList, '/perception/traffic_light', self.traffic_light_callback, 10)
        self.gnss_sub = self.create_subscription(
            NavSatFix, '/carla/ego_vehicle/gnss', self.gnss_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/carla/ego_vehicle/odometry', self.odom_callback, 10)

        # Logging
        self.log_file = open("vehicle_path.log", "w")
        self.log_file.write("timestamp,x,y,z\n")
        self.log_timer = self.create_timer(0.5, self.log_position)

        # Publish initial path
        self.publish_path()
        self.publish_next_waypoint()

    def precompute_distances(self):
        n = len(self.points)
        matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                matrix[i][j] = math.dist(self.points[i], self.points[j])
        return matrix

    def solve_tsp(self):
        initial_mask = 0
        heap = []
        heuristic = self.calculate_heuristic(0, initial_mask)
        heapq.heappush(heap, (heuristic, 0, 0, initial_mask, [0]))
        
        visited = {}
        
        while heap:
            priority, cost, current, mask, path = heapq.heappop(heap)
            
            if mask == (1 << 14) - 1:
                return [self.points[i] for i in path]
            
            state_key = (current, mask)
            if state_key in visited and visited[state_key] <= cost:
                continue
            visited[state_key] = cost
            
            for next_wp in range(14):
                if not (mask & (1 << next_wp)):
                    next_index = next_wp + 1
                    new_cost = cost + self.distance_matrix[current][next_index]
                    new_mask = mask | (1 << next_wp)
                    new_path = path + [next_index]
                    new_heuristic = self.calculate_heuristic(next_index, new_mask)
                    total_priority = new_cost + new_heuristic
                    
                    heapq.heappush(heap, (total_priority, new_cost, next_index, new_mask, new_path))
        
        return None

    def calculate_heuristic(self, current, mask):
        unvisited = [i+1 for i in range(14) if not (mask & (1 << i))]
        if not unvisited:
            return 0
        min_dist = min(self.distance_matrix[current][wp] for wp in unvisited)
        return min_dist + self.mst_cost(unvisited)

    def mst_cost(self, nodes):
        if len(nodes) <= 1:
            return 0
        edges = []
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                edges.append((self.distance_matrix[nodes[i]][nodes[j]], i, j))
        edges.sort()
        parent = list(range(len(nodes)))
        total = 0
        
        def find(u):
            if parent[u] != u:
                parent[u] = find(parent[u])
            return parent[u]
        
        for cost, u, v in edges:
            pu, pv = find(u), find(v)
            if pu != pv:
                parent[pu] = pv
                total += cost
                if len(nodes) - 1 == 0:
                    break
        return total

    def publish_path(self):
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map"
        for point in self.optimal_path:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            pose.pose.position.z = point[2]
            path_msg.poses.append(pose)
        self.path_pub.publish(path_msg)

    def publish_next_waypoint(self):
        if self.current_waypoint_idx < len(self.optimal_path):
            wp = self.optimal_path[self.current_waypoint_idx]
            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = "map"
            pose.pose.position.x = wp[0]
            pose.pose.position.y = wp[1]
            pose.pose.position.z = wp[2]
            self.next_wp_pub.publish(pose)

    def obstacles_callback(self, msg):
        self.obstacles = msg.obstacles

    def lane_callback(self, msg):
        self.lane_detected = msg.lane_detected

    def traffic_light_callback(self, msg):
        self.traffic_light_status = msg.status

    def gnss_callback(self, msg):
        self.gnss_position = (msg.latitude, msg.longitude, msg.altitude)

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose
        current_x = self.current_pose.position.x
        current_y = self.current_pose.position.y
        target = self.optimal_path[self.current_waypoint_idx]
        distance = math.hypot(target[0] - current_x, target[1] - current_y)
        
        # Check for obstacles, lane, and traffic light before proceeding
        if (distance < 2.0 and 
            self.current_waypoint_idx < len(self.optimal_path) - 1 and
            not self.obstacles and 
            self.lane_detected and
            (self.traffic_light_status is None or all(light.state == 0 for light in self.traffic_light_status))):  # 0 for green
            self.current_waypoint_idx += 1
            self.publish_next_waypoint()

    def log_position(self):
        if self.current_pose:
            x = self.current_pose.position.x
            y = self.current_pose.position.y
            z = self.current_pose.position.z
            self.log_file.write(f"{time.time()},{x},{y},{z}\n")

    def __del__(self):
        self.log_file.close()

def main(args=None):
    rclpy.init(args=args)
    node = PlanningNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()