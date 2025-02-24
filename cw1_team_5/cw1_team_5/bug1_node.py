#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose2D
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
import numpy as np
import math

class Bug1Node(Node):
    def __init__(self):
        super().__init__('bug1_node')
        
        # Parameters
        self.SAFETY_DISTANCE = 0.5      # Distance to trigger obstacle avoidance
        self.GOAL_TOLERANCE = 0.2       # Distance at which goal is considered reached
        self.ANGULAR_SPEED = 0.5        # Angular speed for rotations (rad/s)
        self.LINEAR_SPEED = 0.3         # Linear speed for moving forward (m/s)
        self.WALL_FOLLOW_DISTANCE = 0.7 # Preferred distance to keep from the wall
        
        # State variables
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_orientation = 0.0
        self.goal_x = None
        self.goal_y = None
        self.goal_theta = None
        
        # Bug1-specific variables
        self.state = 'IDLE'   # States: IDLE, MOVE_TO_GOAL, WALL_FOLLOWING
        self.hit_point = None
        self.leave_point = None
        self.closest_point = None  # The point (while wall following) closest to the goal
        self.min_distance_to_goal = float('inf')
        self.circumnavigation_start_point = None
        self.complete_circumnavigation = False
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Subscribers (ensure topic names match your simulation)
        self.odom_sub = self.create_subscription(
            Odometry,
            '/Odometry',  # Verify that your odom topic is indeed '/Odometry'
            self.odom_callback,
            10)
            
        self.goal_sub = self.create_subscription(
            Pose2D,
            '/waypoint',  # For Bug1, we use the same type as Lab 3's waypoint (or /goal if specified)
            self.goal_callback,
            10)
            
        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            '/mid360_PointCloud2',
            self.pointcloud_callback,
            10)
            
        # Timer for control loop at 10 Hz
        self.timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info('Bug1 node initialized')

    def odom_callback(self, msg):
        """Update current robot pose from odometry."""
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        
        # Convert quaternion to yaw
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_orientation = math.atan2(siny_cosp, cosy_cosp)

    def goal_callback(self, msg):
        """Handle new goal. Reset Bug1 state variables."""
        self.goal_x = msg.x
        self.goal_y = msg.y
        self.goal_theta = msg.theta
        self.state = 'MOVE_TO_GOAL'
        self.hit_point = None
        self.leave_point = None
        self.closest_point = None
        self.min_distance_to_goal = self.distance_to_goal()
        self.complete_circumnavigation = False
        self.circumnavigation_start_point = None
        self.get_logger().info(f'New goal received: ({self.goal_x}, {self.goal_y})')

    def pointcloud_callback(self, msg):
        """Process pointcloud data to detect obstacles."""
        if self.state == 'IDLE':
            return
            
        # Read points from the pointcloud
        points = list(point_cloud2.read_points(msg, field_names=['x', 'y', 'z']))
        self.process_obstacles(points)

    def process_obstacles(self, points):
        """Process pointcloud data to detect the closest obstacle point."""
        min_distance = float('inf')
        closest_obs_point = None
        
        for point in points:
            x, y = point[0], point[1]
            distance = math.sqrt(x*x + y*y)
            # Ignore points too close to the sensor (noise)
            if distance < 0.1:
                continue
            if distance < min_distance:
                min_distance = distance
                closest_obs_point = (x, y)
                
        # If an obstacle is detected within safety distance while moving toward the goal,
        # switch to wall following.
        if min_distance < self.SAFETY_DISTANCE and self.state == 'MOVE_TO_GOAL':
            self.state = 'WALL_FOLLOWING'
            if self.hit_point is None:
                self.hit_point = (self.current_x, self.current_y)
                self.circumnavigation_start_point = (self.current_x, self.current_y)
                self.get_logger().info('Obstacle detected: switching to WALL_FOLLOWING state')
                
        # While wall following, record the point closest to the goal.
        if self.state == 'WALL_FOLLOWING':
            current_distance = self.distance_to_goal()
            if current_distance < self.min_distance_to_goal:
                self.min_distance_to_goal = current_distance
                self.closest_point = (self.current_x, self.current_y)

    def control_loop(self):
        """Main control loop executing the Bug1 algorithm."""
        # Do nothing if idle or no goal is set.
        if self.state == 'IDLE' or self.goal_x is None:
            return
            
        # Check if goal is reached.
        if self.distance_to_goal() < self.GOAL_TOLERANCE:
            self.state = 'IDLE'
            self.stop_robot()
            self.get_logger().info('Goal reached!')
            return
            
        cmd_vel = Twist()
        
        if self.state == 'MOVE_TO_GOAL':
            self.move_to_goal(cmd_vel)
        elif self.state == 'WALL_FOLLOWING':
            self.wall_following(cmd_vel)
            
        self.cmd_vel_pub.publish(cmd_vel)

    def move_to_goal(self, cmd_vel):
        """Drive directly towards the goal using simple proportional control."""
        angle_to_goal = self.angle_to_goal()
        angle_error = self.normalize_angle(angle_to_goal - self.current_orientation)
        
        # Rotate if the error is significant; else move forward.
        if abs(angle_error) > 0.1:
            cmd_vel.angular.z = self.ANGULAR_SPEED * np.sign(angle_error)
            cmd_vel.linear.x = 0.0
        else:
            cmd_vel.linear.x = self.LINEAR_SPEED
            cmd_vel.angular.z = 0.3 * angle_error

    def wall_following(self, cmd_vel):
        """Perform wall following until a leaving condition is met."""
        # Optional: check if a clear path to the goal exists (this is a more advanced check
        # and could be implemented by processing the point cloud further).

        # Check if we have circumnavigated the obstacle:
        if self.circumnavigation_start_point is not None and not self.complete_circumnavigation:
            dist_to_start = math.sqrt(
                (self.current_x - self.circumnavigation_start_point[0])**2 +
                (self.current_y - self.circumnavigation_start_point[1])**2
            )
            # If close to the start of circumnavigation and a hit point exists,
            # assume we've completed the boundary and can leave.
            if dist_to_start < 0.3 and self.hit_point is not None:
                self.complete_circumnavigation = True
                # Set the leave point to the recorded closest point.
                self.leave_point = self.closest_point
                self.get_logger().info('Circumnavigation complete. Leaving wall and moving to goal.')
                self.state = 'MOVE_TO_GOAL'
                return

        # Basic wall following command: adjust as needed for smoother behavior.
        # This simplistic controller maintains a forward velocity with a constant angular component.
        cmd_vel.linear.x = self.LINEAR_SPEED * 0.5
        cmd_vel.angular.z = self.ANGULAR_SPEED * 0.5

    def distance_to_goal(self):
        """Calculate Euclidean distance to the goal."""
        if self.goal_x is None:
            return float('inf')
        return math.sqrt((self.goal_x - self.current_x)**2 + (self.goal_y - self.current_y)**2)

    def angle_to_goal(self):
        """Calculate angle from current pose to goal."""
        if self.goal_x is None:
            return 0.0
        return math.atan2(self.goal_y - self.current_y, self.goal_x - self.current_x)

    def stop_robot(self):
        """Publish zero velocity to stop the robot."""
        cmd_vel = Twist()
        self.cmd_vel_pub.publish(cmd_vel)

    def normalize_angle(self, angle):
        """Normalize an angle to the range [-pi, pi]."""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

def main(args=None):
    rclpy.init(args=args)
    node = Bug1Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
