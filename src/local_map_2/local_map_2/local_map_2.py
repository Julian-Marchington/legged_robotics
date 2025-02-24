#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import numpy as np

class LocalMapCreatorNode(Node):
    def __init__(self):
        super().__init__('local_map_creator_node')
        
        # Parameters
        self.min_distance = 0.3
        self.max_distance = 3.0
        self.angle_bins = 180
        self.max_angle_diff = np.radians(15)
        self.max_point_distance = 0.2
        self.min_points_per_bin = 5
        
        # Subscriber for point cloud
        self.subscription = self.create_subscription(
            PointCloud2,
            'mid360_PointCloud2',
            self.point_cloud_callback,
            10)
        
        # Publishers for visualization
        self.points_pub = self.create_publisher(Marker, 'local_map_points', 10)
        self.lines_pub = self.create_publisher(Marker, 'local_map_lines', 10)

        self.get_logger().info('Local map creator node initialized')

    def filter_and_bin_points(self, points):
        """Filter points by distance and bin them by angle"""
        angle_bins = [[] for _ in range(self.angle_bins)]
        bin_size = 2 * np.pi / self.angle_bins
        
        for point in points:
            # Extract x, y. We ignore z as we are creating a 2D map
            x, y, _ = point
            distance = np.sqrt(x*x + y*y)
            
            if self.min_distance < distance < self.max_distance:
                angle = np.arctan2(y, x)

                if angle < 0:
                    angle += 2 * np.pi
                    
                bin_idx = int(angle / bin_size)
                if bin_idx == self.angle_bins:
                    bin_idx = 0
                    
                angle_bins[bin_idx].append([x, y, distance])
        
        return angle_bins

    def average_bins(self, angle_bins):
        """Average points in each non-empty bin with outlier rejection"""
        averaged_points = []
        bin_indices = []
        
        for i, bin_points in enumerate(angle_bins):
            if len(bin_points) >= self.min_points_per_bin:
                bin_array = np.array(bin_points)
                
                # Get distances for outlier rejection
                distances = bin_array[:, 2]
                
                # Reject outliers based on distance
                mean_dist = np.mean(distances)
                std_dist = np.std(distances)
                mask = np.abs(distances - mean_dist) <= 2 * std_dist
                
                if np.sum(mask) >= self.min_points_per_bin:
                    # Average only non-outlier points
                    filtered_points = bin_array[mask][:, :2]
                    avg_point = np.mean(filtered_points, axis=0)
                    averaged_points.append(avg_point)
                    bin_indices.append(i)
        
        return np.array(averaged_points), np.array(bin_indices)

    def create_line_segments(self, points, bin_indices):
        """Create line segments between nearby points"""
    ## Recommend removing this for tutorial
    # --------------------------------------------------#
        if len(points) < 2:
            return []
            
        segments = []
        current_segment = []
        
        for i in range(len(points)-1):
            current_point = points[i]
            next_point = points[i+1]
            
            # Calculate angular difference
            bin_diff = abs(bin_indices[i+1] - bin_indices[i])
            if bin_diff > self.angle_bins/2:
                bin_diff = self.angle_bins - bin_diff
            angle_diff = bin_diff * (2 * np.pi / self.angle_bins)
            
            # Calculate spatial distance between points
            distance = np.linalg.norm(next_point - current_point)
            
            # Check if points should be connected
            if angle_diff < self.max_angle_diff and distance < self.max_point_distance:
                if not current_segment:
                    current_segment.append(current_point)
                current_segment.append(next_point)
            else:
                # End current segment if it has at least 2 points
                if len(current_segment) >= 2:
                    # Check for minimum segment length
                    segment_length = np.linalg.norm(current_segment[-1] - current_segment[0])

                    # Add segment if it is longer than 30cm (we are assuming walls are at least 30cm wide)
                    if segment_length > 0.3:
                        segments.append(current_segment)
                current_segment = []
        
        # Add final segment if it exists and meets length requirement
        if len(current_segment) >= 2:
            segment_length = np.linalg.norm(current_segment[-1] - current_segment[0])
            if segment_length > 0.3:
                segments.append(current_segment)
        # -------------------------------------------------- #
        return segments

    def publish_visualization(self, points, segments):
        # Publish points
        point_marker = Marker()
        point_marker.header.frame_id = "livox"
        point_marker.header.stamp = self.get_clock().now().to_msg()
        point_marker.type = Marker.POINTS
        point_marker.action = Marker.ADD
        point_marker.id = 0
        
        point_marker.scale.x = 0.05
        point_marker.scale.y = 0.05
        point_marker.color.r = 1.0
        point_marker.color.a = 1.0
        
        for point in points:
            p = Point()
            p.x = float(point[0])
            p.y = float(point[1])
            p.z = 0.0
            point_marker.points.append(p)
        
        self.points_pub.publish(point_marker)
        
        # Publish line segments
        line_marker = Marker()
        line_marker.header.frame_id = "livox"
        line_marker.header.stamp = self.get_clock().now().to_msg()
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.id = 0
        
        line_marker.scale.x = 0.03
        line_marker.color.g = 1.0
        line_marker.color.a = 1.0
        
        # Create separate line strips for each segment
        for i, segment in enumerate(segments):
            line_marker = Marker()
            line_marker.header.frame_id = "livox"
            line_marker.header.stamp = self.get_clock().now().to_msg()
            line_marker.type = Marker.LINE_STRIP
            line_marker.action = Marker.ADD
            line_marker.id = i
            
            line_marker.scale.x = 0.03
            line_marker.color.g = 1.0
            line_marker.color.a = 1.0
            
            for point in segment:
                p = Point()
                p.x = float(point[0])
                p.y = float(point[1])
                p.z = 0.0
                line_marker.points.append(p)
            
            self.lines_pub.publish(line_marker)

    def point_cloud_callback(self, msg):
        # Convert ROS PointCloud2 to numpy array
        points = []
        for p in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            points.append([p[0], p[1], p[2]])
            
        if len(points) < 10:
            return
            
        # Process points
        angle_bins = self.filter_and_bin_points(points)
        averaged_points, bin_indices = self.average_bins(angle_bins)
        
        if len(averaged_points) < 2:
            return
            
        # Create line segments
        segments = self.create_line_segments(averaged_points, bin_indices)
        
        # Publish visualization
        self.publish_visualization(averaged_points, segments)

def main(args=None):
    rclpy.init(args=args)
    node = LocalMapCreatorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


