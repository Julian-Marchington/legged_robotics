#!/usr/bin/env python3

import math
import numpy as np
import rclpy
from rclpy.node import Node
from collections import deque

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose2D, Point, Vector3
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA


class EdgeFollowerNode(Node):
    def __init__(self):
        super().__init__('edge_follower_node')

        # Constants
        self.SAFETY_MARGIN = 0.8        # meters offset from the edge
        self.INCREMENT_DISTANCE = 0.8   # smaller steps to reduce large jumps
        self.UPDATE_RATE = 0.5          # seconds

        # State variables
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_orientation = 0.0
        self.is_odom_received = False

        # Current edges from the local map (list of (start, end) pairs, each a 2D np.array)
        self.current_edges = []

        # (Optional) Rolling buffer of waypoints for smoothing
        self.last_waypoints = deque()
        self.waypoint_window_size = 5   # average over last N waypoints

        # Publishers
        self.waypoint_pub = self.create_publisher(Pose2D, 'waypoint', 10)
        self.waypoint_marker_pub = self.create_publisher(Marker, 'current_waypoint', 10)
        self.edge_marker_pub = self.create_publisher(Marker, 'detected_edges', 10)

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry,
            'Odometry',
            self.odom_callback,
            10
        )
        self.line_sub = self.create_subscription(
            Marker,
            'local_map_lines',
            self.line_callback,
            10
        )

        # Timer
        self.timer = self.create_timer(self.UPDATE_RATE, self.timer_callback)

        self.get_logger().info('Edge Follower node initialized (CCW + smoothing)')

    def odom_callback(self, msg):
        """Update current robot pose (x,y,theta) from odometry (in odom frame)."""
        self.is_odom_received = True
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

        # Convert quaternion to yaw (theta)
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_orientation = math.atan2(siny_cosp, cosy_cosp)

    def line_callback(self, msg):
        """Process incoming line segments (edges) from the local map."""
        if len(msg.points) < 2:
            return

        # Convert line points to numpy arrays
        points = [np.array([p.x, p.y]) for p in msg.points]

        # Create edges by connecting adjacent points
        self.current_edges = []
        for i in range(len(points) - 1):
            self.current_edges.append((points[i], points[i + 1]))

    def transform_to_camera_init(self, point):
        """
        Transform a point from the robot's base frame (livox) to a fixed world frame (camera_init).
        This is typically used before publishing a waypoint in the global coordinate frame.
        """
        c = math.cos(self.current_orientation)
        s = math.sin(self.current_orientation)

        # The robot is at (self.current_x, self.current_y) in camera_init,
        # so we rotate then translate.
        x = point[0] * c - point[1] * s + self.current_x
        y = point[0] * s + point[1] * c + self.current_y
        return np.array([x, y])

    def transform_to_base_link(self, point):
        """
        Transform a point from the camera_init frame to the robot's base frame (livox).
        """
        dx = point[0] - self.current_x
        dy = point[1] - self.current_y
        c = math.cos(-self.current_orientation)
        s = math.sin(-self.current_orientation)

        x = dx * c - dy * s
        y = dx * s + dy * c
        return np.array([x, y])

    def find_next_waypoint(self):
        """
        Find the closest point on any edge, then move 'forward' along the edge by
        self.INCREMENT_DISTANCE, and finally offset to the left (counter-clockwise).
        """
        if not self.current_edges:
            return None

        # The robot is at (0,0) in its own base frame.
        robot_pos = np.array([0.0, 0.0])

        # 1) Find the closest point on any edge
        closest_edge = None
        closest_point = None
        min_distance = float('inf')
        closest_edge_index = 0

        for i, edge in enumerate(self.current_edges):
            start_point, end_point = edge
            edge_vector = end_point - start_point
            edge_length = np.linalg.norm(edge_vector)

            # Skip very short edges
            if edge_length < 1e-3:
                continue

            edge_direction = edge_vector / edge_length
            to_robot = robot_pos - start_point
            projection = np.dot(to_robot, edge_direction)
            # Clamp projection to the segment
            projection = max(0, min(edge_length, projection))

            point_on_edge = start_point + projection * edge_direction
            distance = np.linalg.norm(point_on_edge - robot_pos)

            if distance < min_distance:
                min_distance = distance
                closest_edge = edge
                closest_point = point_on_edge
                closest_edge_index = i

        if closest_edge is None:
            return None

        # Save for visualization
        self.closest_edge_point = closest_point

        # 2) Move forward along the edges by INCREMENT_DISTANCE
        current_index = closest_edge_index
        increment_left = self.INCREMENT_DISTANCE
        current_point = closest_point

        while increment_left > 0 and current_index < len(self.current_edges):
            start, end = self.current_edges[current_index]
            segment_length = np.linalg.norm(end - start)

            # Distance from current_point to the end of this segment
            remaining = np.linalg.norm(end - current_point)

            if increment_left <= remaining:
                direction = (end - start) / segment_length
                current_point = current_point + direction * increment_left
                increment_left = 0
            else:
                increment_left -= remaining
                current_index += 1
                if current_index < len(self.current_edges):
                    current_point = self.current_edges[current_index][0]
                else:
                    current_point = self.current_edges[-1][1]
                    break

        # Save for visualization
        self.incremented_point = current_point

        # 3) Compute the CCW offset from the edge
        # Determine the edge direction for the current segment
        if current_index < len(self.current_edges):
            start, end = self.current_edges[current_index]
        else:
            start, end = self.current_edges[-1]

        edge_direction = end - start
        edge_length = np.linalg.norm(edge_direction)
        if edge_length < 1e-3:
            # If something is degenerate, just skip
            return None

        edge_direction /= edge_length  # normalize

        # For an ideal CCW offset, we need to figure out which side is "left" relative to the robot
        # We'll do a cross product of (edge_direction) and the vector from current_point to the robot
        to_robot = robot_pos - current_point
        cross_z = edge_direction[0] * to_robot[1] - edge_direction[1] * to_robot[0]

        # The left perpendicular is [-y, x]
        left_perp = np.array([-edge_direction[1], edge_direction[0]])

        # If cross_z < 0, robot is on one side; if cross_z > 0, it's on the other.
        # We want the offset that keeps the obstacle on the left side from the robot's perspective.
        if cross_z < 0:
            offset_vector = left_perp
        else:
            # Flip it if the robot is on the other side
            offset_vector = -left_perp

        # The final waypoint is offset by SAFETY_MARGIN
        waypoint = current_point + offset_vector * self.SAFETY_MARGIN
        return waypoint

    def publish_visualizations(self, current_waypoint):
        """
        Publish visualization markers for the current waypoint, closest edge point, etc.
        """
        # Waypoint marker (blue sphere)
        if current_waypoint is not None:
            waypoint_marker = Marker()
            waypoint_marker.header.frame_id = "livox"
            waypoint_marker.header.stamp = self.get_clock().now().to_msg()
            waypoint_marker.type = Marker.SPHERE
            waypoint_marker.action = Marker.ADD
            waypoint_marker.id = 0
            waypoint_marker.pose.position.x = float(current_waypoint[0])
            waypoint_marker.pose.position.y = float(current_waypoint[1])
            waypoint_marker.pose.position.z = 0.0
            waypoint_marker.scale = Vector3(x=0.2, y=0.2, z=0.2)
            waypoint_marker.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0)
            self.waypoint_marker_pub.publish(waypoint_marker)

            # Closest point on edge (red sphere)
            if hasattr(self, 'closest_edge_point'):
                red_marker = Marker()
                red_marker.header.frame_id = "livox"
                red_marker.header.stamp = self.get_clock().now().to_msg()
                red_marker.type = Marker.SPHERE
                red_marker.action = Marker.ADD
                red_marker.id = 1
                red_marker.pose.position.x = float(self.closest_edge_point[0])
                red_marker.pose.position.y = float(self.closest_edge_point[1])
                red_marker.pose.position.z = 0.0
                red_marker.scale = Vector3(x=0.1, y=0.1, z=0.1)
                red_marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
                self.waypoint_marker_pub.publish(red_marker)

            # Incremented point (green sphere)
            if hasattr(self, 'incremented_point'):
                green_marker = Marker()
                green_marker.header.frame_id = "livox"
                green_marker.header.stamp = self.get_clock().now().to_msg()
                green_marker.type = Marker.SPHERE
                green_marker.action = Marker.ADD
                green_marker.id = 2
                green_marker.pose.position.x = float(self.incremented_point[0])
                green_marker.pose.position.y = float(self.incremented_point[1])
                green_marker.pose.position.z = 0.0
                green_marker.scale = Vector3(x=0.1, y=0.1, z=0.1)
                green_marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
                self.waypoint_marker_pub.publish(green_marker)

        # Publish detected edges (yellow lines)
        edge_marker = Marker()
        edge_marker.header.frame_id = "livox"
        edge_marker.header.stamp = self.get_clock().now().to_msg()
        edge_marker.type = Marker.LINE_LIST
        edge_marker.action = Marker.ADD
        edge_marker.id = 3
        edge_marker.scale.x = 0.05
        edge_marker.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)

        for start_point, end_point in self.current_edges:
            edge_marker.points.append(Point(x=float(start_point[0]), y=float(start_point[1]), z=0.0))
            edge_marker.points.append(Point(x=float(end_point[0]), y=float(end_point[1]), z=0.0))

        self.edge_marker_pub.publish(edge_marker)

    def timer_callback(self):
        """
        Main control loop. Compute the next waypoint, apply smoothing, then publish.
        """
        if not self.is_odom_received:
            return

        # Compute a new waypoint in the robot's base frame
        next_waypoint = self.find_next_waypoint()
        if next_waypoint is None:
            # No edges or no valid waypoint
            return

        # (Optional) Smooth the waypoint by averaging over the last N
        self.last_waypoints.append(next_waypoint)
        if len(self.last_waypoints) > self.waypoint_window_size:
            self.last_waypoints.popleft()

        # Convert deque to array, then average
        waypoints_array = np.array(self.last_waypoints)
        smoothed_waypoint = waypoints_array.mean(axis=0)

        # Transform to camera_init frame for publishing
        waypoint_camera_init = self.transform_to_camera_init(smoothed_waypoint)

        # Publish the final waypoint
        waypoint_msg = Pose2D()
        waypoint_msg.x = float(waypoint_camera_init[0])
        waypoint_msg.y = float(waypoint_camera_init[1])
        # Keep the current orientation or compute a desired heading if needed
        waypoint_msg.theta = self.current_orientation
        self.waypoint_pub.publish(waypoint_msg)

        # For visualization, show the smoothed waypoint in base frame
        self.publish_visualizations(smoothed_waypoint)


def main(args=None):
    rclpy.init(args=args)
    node = EdgeFollowerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt, shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
