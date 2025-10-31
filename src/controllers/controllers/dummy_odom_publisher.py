#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
import math

class DummyOdomPublisher(Node):
    def __init__(self):
        super().__init__('dummy_odom_publisher')
        
        # Publisher for odometry
        self.odom_pub = self.create_publisher(Odometry, '/odometry/filtered', 10)
        
        # Timer to publish odometry at high frequency
        self.timer = self.create_timer(0.02, self.publish_odom)  # 50 Hz
        
        self.get_logger().info("Dummy Odometry Publisher initialized - publishing at 50Hz")

    def publish_odom(self):
        """Publish a stationary robot odometry for testing."""
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "Chassis"
        
        # Robot at origin, stationary
        odom_msg.pose.pose.position.x = 0.0
        odom_msg.pose.pose.position.y = 0.0
        odom_msg.pose.pose.position.z = 0.0
        
        # No rotation
        odom_msg.pose.pose.orientation.w = 1.0
        odom_msg.pose.pose.orientation.x = 0.0
        odom_msg.pose.pose.orientation.y = 0.0
        odom_msg.pose.pose.orientation.z = 0.0
        
        # Zero velocity
        odom_msg.twist.twist.linear.x = 0.0
        odom_msg.twist.twist.linear.y = 0.0
        odom_msg.twist.twist.linear.z = 0.0
        odom_msg.twist.twist.angular.x = 0.0
        odom_msg.twist.twist.angular.y = 0.0
        odom_msg.twist.twist.angular.z = 0.0
        
        # Add some covariance (optional)
        odom_msg.pose.covariance[0] = 0.1  # x
        odom_msg.pose.covariance[7] = 0.1  # y
        odom_msg.pose.covariance[35] = 0.1  # yaw
        
        odom_msg.twist.covariance[0] = 0.1  # linear.x
        odom_msg.twist.covariance[35] = 0.1  # angular.z
        
        self.odom_pub.publish(odom_msg)


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = DummyOdomPublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
