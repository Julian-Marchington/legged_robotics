#include <rclcpp/rclcpp.hpp>
#include "unitree_api/msg/request.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "common/ros2_sport_client.h"

using std::placeholders::_1;

class soprt_request : public rclcpp::Node
{
public:
    soprt_request() : Node("req_sender")
    {
        // Create subscriber for cmd_vel
        cmd_vel_suber = this->create_subscription<geometry_msgs::msg::Twist>(
            "/cmd_vel", 10, std::bind(&soprt_request::cmd_vel_callback, this, _1));
        // Create publisher for sport request
        req_puber = this->create_publisher<unitree_api::msg::Request>("/api/sport/request", 10);
        std::cout << "Forward Cmd_Vel from Follower to GO2 Movement Command" << std::endl;
    };

private:
    void cmd_vel_callback(const geometry_msgs::msg::Twist::SharedPtr msg)
    {
        // Extract velocities from the message
        float linear_x = msg->linear.x;
        float linear_y = msg->linear.y;
        float angular_z = msg->angular.z;

        // Check if all velocities are zero
        if (linear_x == 0 && linear_y == 0 && angular_z == 0) {
            std::cout << "Stop" << std::endl;
            sport_req.StopMove(req);
        } else {
            std::cout << "Moving" << std::endl;
            sport_req.Move(req, linear_x, linear_y, angular_z);
        }

        // Publish the request
        req_puber->publish(req);
    }

    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_suber;
    rclcpp::Publisher<unitree_api::msg::Request>::SharedPtr req_puber;
    unitree_api::msg::Request req;
    SportClient sport_req;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<soprt_request>());
    rclcpp::shutdown();
    return 0;
}