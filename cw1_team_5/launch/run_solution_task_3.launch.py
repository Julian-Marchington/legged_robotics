from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='cw1_team_5',
            executable='bug1_node',
            name='bug1_node',
            output='screen',
            parameters=[{'use_sim_time': True}]
        )
    ]) 
