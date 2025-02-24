#!/usr/bin/env python3

import math
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose2D, Point, Vector3
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA

class Bug0Node(Node):
    def __init__(self):
        super().__init__('bug0_node')
        
        # Constants
        self.SAFETY_MARGIN = 1.0       # Distance from obstacles (meters)
        self.INCREMENT_DISTANCE = 0.7  # Step size along boundary or toward goal (meters)
        self.CORNER_OFFSET = 0.3       # Additional offset for corner turning (meters)
        self.UPDATE_RATE = 0.5         # Waypoint update frequency (seconds)
        self.GOAL_TOLERANCE = 0.5      # Distance to consider goal reached (meters)
        
        # State variables
        self.state = 'GO_TO_GOAL'      # Initial state
        self.current_x = 0.0           # Robot x-position in camera_init frame
        self.current_y = 0.0           # Robot y-position in camera_init frame
        self.current_orientation = 0.0 # Robot yaw in camera_init frame
        self.is_odom_received = False
        self.goal = None               # Goal position in camera_init frame [x, y]
        self.current_edges = []        # List of (start_point, end_point) tuples in livox frame
        
        # Publishers
        self.waypoint_pub = self.create_publisher(Pose2D, 'waypoint', 10)
        self.waypoint_marker_pub = self.create_publisher(Marker, 'current_waypoint', 10)
        self.edge_marker_pub = self.create_publisher(Marker, 'detected_edges', 10)
        
        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry,
            'Odometry',
            self.odom_callback,
            10)
        self.goal_sub = self.create_subscription(
            Pose2D,
            'goal',
            self.goal_callback,
            10)
        self.line_sub = self.create_subscription(
            Marker,
            'local_map_lines',
            self.line_callback,
            10)
            
        # Timer for periodic updates
        self.timer = self.create_timer(self.UPDATE_RATE, self.timer_callback)
        
        self.get_logger().info('Bug0 node initialized')

    def odom_callback(self, msg):
        """Update robot's current pose from odometry in camera_init frame."""
        self.is_odom_received = True
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        
        # Convert quaternion to yaw
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_orientation = math.atan2(siny_cosp, cosy_cosp)

    def goal_callback(self, msg):
        """Store the goal position from the /goal topic in camera_init frame."""
        self.goal = np.array([msg.x, msg.y])
        self.get_logger().info(f'Goal received: x={msg.x}, y={msg.y}')

    def transform_to_camera_init(self, point):
        """Transform a point from livox frame to camera_init frame."""
        c = math.cos(self.current_orientation)
        s = math.sin(self.current_orientation)
        x = point[0] * c - point[1] * s + self.current_x
        y = point[0] * s + point[1] * c + self.current_y
        return np.array([x, y])

    def transform_to_base_link(self, point):
        """Transform a point from camera_init frame to livox frame."""
        dx = point[0] - self.current_x
        dy = point[1] - self.current_y
        c = math.cos(-self.current_orientation)
        s = math.sin(-self.current_orientation)
        x = dx * c - dy * s
        y = dx * s + dy * c
        return np.array([x, y])
    
    def line_callback(self, msg):
        """Process edge segments from local_map_lines in livox frame."""
        if len(msg.points) < 2:
            self.current_edges = []
            return
        points = [np.array([point.x, point.y]) for point in msg.points]
        self.current_edges = [(points[i], points[i+1]) for i in range(len(points) - 1)]

    def ccw(self, A, B, C):
        """Check if points A, B, C are in counterclockwise order."""
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def line_segment_intersection(self, A, B, C, D):
        """Check if line segments AB and CD intersect."""
        return (self.ccw(A, C, D) != self.ccw(B, C, D) and 
                self.ccw(A, B, C) != self.ccw(A, B, D))

    def line_intersects_edges(self, start, end, edges):
        """Check if line from start to end intersects any edge segment."""
        for edge in edges:
            if self.line_segment_intersection(start, end, edge[0], edge[1]):
                return True
        return False

    def find_next_waypoint(self):
        """Calculate the next waypoint along the obstacle boundary in livox frame."""
        if not self.current_edges:
            self.get_logger().warn('No edges detected for boundary following')
            return None
            
        robot_pos = np.array([0.0, 0.0])  # Robot position in livox frame
        
        # Find closest point on any edge
        min_distance = float('inf')
        closest_point = None
        closest_edge = None
        closest_edge_index = 0
        
        for i, edge in enumerate(self.current_edges):
            start_point, end_point = edge
            edge_vector = end_point - start_point
            edge_length = np.linalg.norm(edge_vector)
            
            if edge_length < 0.01:
                continue
                
            edge_direction = edge_vector / edge_length
            to_robot = robot_pos - start_point
            projection = np.dot(to_robot, edge_direction)
            projection = max(0, min(edge_length, projection))
            point_on_edge = start_point + projection * edge_direction
            distance = np.linalg.norm(point_on_edge - robot_pos)
            
            if distance < min_distance:
                min_distance = distance
                closest_point = point_on_edge
                closest_edge = edge
                closest_edge_index = i
        
        if closest_edge is None:
            self.get_logger().warn('No valid closest edge found')
            return None

        self.closest_edge_point = closest_point
        start_point, end_point = closest_edge
        edge_vector = end_point - start_point
        edge_direction = edge_vector / np.linalg.norm(edge_vector)
        to_robot = robot_pos - closest_point
        cross_z = edge_direction[0] * to_robot[1] - edge_direction[1] * to_robot[0]
        moving_forward = cross_z > 0  # Direction based on robot's position
        
        # Move along edges by INCREMENT_DISTANCE
        current_index = closest_edge_index
        increment_left = self.INCREMENT_DISTANCE
        current_point = closest_point
        
        if moving_forward:
            while increment_left > 0 and current_index < len(self.current_edges) - 1:
                current_edge = self.current_edges[current_index]
                start, end = current_edge
                remaining_distance = np.linalg.norm(end - current_point)
                
                if increment_left <= remaining_distance:
                    edge_direction = (end - start) / np.linalg.norm(end - start)
                    current_point = current_point + edge_direction * increment_left
                    increment_left = 0
                    break
                else:
                    increment_left -= remaining_distance
                    current_index += 1
                    if current_index < len(self.current_edges):
                        current_point = self.current_edges[current_index][0]
                    else:
                        current_point = self.current_edges[-1][1]
                        break
        else:
            while increment_left > 0 and current_index > 0:
                current_edge = self.current_edges[current_index]
                start, end = current_edge
                remaining_distance = np.linalg.norm(current_point - start)
                
                if increment_left <= remaining_distance:
                    edge_direction = (end - start) / np.linalg.norm(end - start)
                    current_point = current_point - edge_direction * increment_left
                    increment_left = 0
                    break
                else:
                    increment_left -= remaining_distance
                    current_index -= 1
                    if current_index >= 0:
                        current_point = self.current_edges[current_index][1]
                    else:
                        current_point = self.current_edges[0][0]
                        break

        # If we exited the loop without breaking, use the last valid point
        if increment_left > 0:
            current_index = min(max(current_index, 0), len(self.current_edges) - 1)

        self.incremented_point = current_point
        current_edge = self.current_edges[current_index]
        start, end = current_edge
        edge_direction = (end - start) / np.linalg.norm(end - start)
        perpendicular = np.array([-edge_direction[1], edge_direction[0]])
        to_robot = robot_pos - current_point
        if np.dot(perpendicular, to_robot) < 0:
            perpendicular = -perpendicular

        tangent_offset = edge_direction if moving_forward else -edge_direction
        waypoint = current_point + perpendicular * self.SAFETY_MARGIN + tangent_offset * self.CORNER_OFFSET
        return waypoint

    def publish_visualizations(self, current_waypoint):
        """Publish markers for waypoints and edges."""
        if current_waypoint is not None:
            marker = Marker()
            marker.header.frame_id = "livox"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.id = 0
            marker.pose.position.x = float(current_waypoint[0])
            marker.pose.position.y = float(current_waypoint[1])
            marker.pose.position.z = 0.0
            marker.scale = Vector3(x=0.2, y=0.2, z=0.2)
            marker.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0)
            self.waypoint_marker_pub.publish(marker)
            
            if hasattr(self, 'closest_edge_point'):
                marker.id = 1
                marker.pose.position.x = float(self.closest_edge_point[0])
                marker.pose.position.y = float(self.closest_edge_point[1])
                marker.scale = Vector3(x=0.1, y=0.1, z=0.1)
                marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
                self.waypoint_marker_pub.publish(marker)
                
            if hasattr(self, 'incremented_point'):
                marker.id = 2
                marker.pose.position.x = float(self.incremented_point[0])
                marker.pose.position.y = float(self.incremented_point[1])
                marker.scale = Vector3(x=0.1, y=0.1, z=0.1)
                marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
                self.waypoint_marker_pub.publish(marker)
        
        edge_marker = Marker()
        edge_marker.header.frame_id = "livox"
        edge_marker.header.stamp = self.get_clock().now().to_msg()
        edge_marker.type = Marker.LINE_LIST
        edge_marker.action = Marker.ADD
        edge_marker.id = 0
        edge_marker.scale.x = 0.05
        edge_marker.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)
        
        for start_point, end_point in self.current_edges:
            edge_marker.points.append(Point(x=float(start_point[0]), y=float(start_point[1]), z=0.0))
            edge_marker.points.append(Point(x=float(end_point[0]), y=float(end_point[1]), z=0.0))
            
        self.edge_marker_pub.publish(edge_marker)

    def timer_callback(self):
        """Main control loop implementing Bug0 algorithm."""
        if not self.is_odom_received or self.goal is None:
            self.get_logger().debug('Waiting for odometry or goal')
            return
            
        current_pos = np.array([self.current_x, self.current_y])
        dist_to_goal = np.linalg.norm(self.goal - current_pos)
        
        if dist_to_goal < self.GOAL_TOLERANCE:
            self.get_logger().info('Goal reached')
            return
            
        goal_livox = self.transform_to_base_link(self.goal)
        robot_pos_livox = np.array([0.0, 0.0])
        
        if self.state == 'GO_TO_GOAL':
            if not self.current_edges or not self.line_intersects_edges(robot_pos_livox, goal_livox, self.current_edges):
                # Path to goal is clear, move toward goal
                vec = self.goal - current_pos
                dist = np.linalg.norm(vec)
                direction = vec / dist
                waypoint = current_pos + direction * min(self.INCREMENT_DISTANCE, dist)
                waypoint_livox = None  # No boundary waypoint in this state
            else:
                # Obstacle detected, switch to boundary following
                self.state = 'BOUNDARY_FOLLOWING'
                waypoint_livox = self.find_next_waypoint()
                waypoint = self.transform_to_camera_init(waypoint_livox) if waypoint_livox is not None else None
        else:  # BOUNDARY_FOLLOWING
            waypoint_livox = self.find_next_waypoint()
            if waypoint_livox is not None:
                waypoint = self.transform_to_camera_init(waypoint_livox)
                # Check if path to goal is clear
                if not self.line_intersects_edges(robot_pos_livox, goal_livox, self.current_edges):
                    self.state = 'GO_TO_GOAL'
                    self.get_logger().info('Path to goal clear, switching to GO_TO_GOAL')
            else:
                waypoint = None
                self.get_logger().warn('No valid waypoint found in boundary following')
        
        if waypoint is not None:
            waypoint_msg = Pose2D()
            waypoint_msg.x = float(waypoint[0])
            waypoint_msg.y = float(waypoint[1])
            waypoint_msg.theta = self.current_orientation  # Maintain current orientation
            self.waypoint_pub.publish(waypoint_msg)
        
        self.publish_visualizations(waypoint_livox if self.state == 'BOUNDARY_FOLLOWING' else None)

def main(args=None):
    rclpy.init(args=args)
    node = Bug0Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt, shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
