import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import cv2
import numpy as np
import message_filters
from typing import Dict
from shepherd_ai.shepherd import Shepherd
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Header

class LiveVideoProcessor(Node):
    def __init__(self):
        super().__init__('live_video_processor')
        
        # Initialize CvBridge
        self.bridge = CvBridge()
        
        # Initialize Shepherd with fixed query
        self.shepherd = Shepherd(query="food")
        
        # Create synchronized subscribers
        self.image_sub = message_filters.Subscriber(
            self,
            Image,
            '/zed/zed_node/rgb/image_rect_color'
        )
        
        self.depth_sub = message_filters.Subscriber(
            self,
            Image,
            '/zed/zed_node/depth/depth_registered'
        )
        
        self.pose_sub = message_filters.Subscriber(
            self,
            Odometry,
            '/odometry/local'
        )
        
        # Create synchronizer
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.depth_sub, self.pose_sub],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.synchronized_callback)
        
        # Frame processing rate control
        self.frame_skip = 2
        self.frame_count = 0
        
        # Add window configuration for detection visualization
        cv2.namedWindow('Object Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Object Detection', 1344, 376)
        
        # Add point cloud publisher
        self.point_cloud_pub = self.create_publisher(
            PointCloud2,
            '/shepherd/object_positions',
            10
        )
        
        # Add timer for publishing point cloud
        self.create_timer(1.0, self.publish_point_cloud)

    def pose_to_dict(self, pose_msg: Odometry) -> Dict:
        """Convert Odometry to dictionary"""
        return {
            'x': pose_msg.pose.pose.position.x,
            'y': pose_msg.pose.pose.position.y,
            'z': pose_msg.pose.pose.position.z,
            'qx': pose_msg.pose.pose.orientation.x,
            'qy': pose_msg.pose.pose.orientation.y,
            'qz': pose_msg.pose.pose.orientation.z,
            'qw': pose_msg.pose.pose.orientation.w
        }

    def synchronized_callback(self, image_msg: Image, depth_msg: Image, pose_msg: Odometry):
        """Process synchronized messages"""
        self.frame_count += 1
        if self.frame_count % self.frame_skip != 0:
            return

        try:
            # Convert ROS messages to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
            
            # Convert pose message to dictionary
            pose_dict = self.pose_to_dict(pose_msg)
            
            # Process frame using Shepherd
            results = self.shepherd.process_frame(cv_image, cv_depth, pose_dict)
            
            # Display detection visualization
            frame_viz = cv2.resize(results['visualization'], (1344, 376))
            cv2.imshow('Object Detection', frame_viz)
            key = cv2.waitKey(1)
            
            # Add keyboard controls
            if key == ord('q'):
                self.destroy_node()
                rclpy.shutdown()
                cv2.destroyAllWindows()
            
        except Exception as e:
            self.get_logger().error(f'Error processing frame: {str(e)}')

    def publish_point_cloud(self):
        """Publish detected object positions as PointCloud2 with similarity intensity"""
        # Get all object positions, similarities, and labels from database
        positions, similarities, labels = self.shepherd.db.get_all_positions_and_similarities()
        
        if len(positions) == 0:
            return
            
        # Create header
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'map'
        
        # Create point cloud fields including intensity
        fields = [
            pc2.PointField(name='x', offset=0, datatype=pc2.PointField.FLOAT32, count=1),
            pc2.PointField(name='y', offset=4, datatype=pc2.PointField.FLOAT32, count=1),
            pc2.PointField(name='z', offset=8, datatype=pc2.PointField.FLOAT32, count=1),
            pc2.PointField(name='intensity', offset=12, datatype=pc2.PointField.FLOAT32, count=1),
        ]
        
        # Normalize similarities to [0, 1] range for intensity
        if len(similarities) > 0:
            min_sim = similarities.min()
            max_sim = similarities.max()
            if max_sim > min_sim:
                normalized_similarities = (similarities - min_sim) / (max_sim - min_sim)
            else:
                normalized_similarities = np.zeros_like(similarities)
        else:
            normalized_similarities = np.array([])
        
        # Create points with intensity
        points = np.column_stack((positions, normalized_similarities))
        
        # Create point cloud
        pc2_msg = pc2.create_cloud(header, fields, points)
        
        # Print debug information
        for pos, sim, label in zip(positions, similarities, labels):
            self.get_logger().info(f"Object: {label} at position: {pos}, similarity: {sim:.3f}")
        
        # Publish
        self.point_cloud_pub.publish(pc2_msg)

def main(args=None):
    rclpy.init(args=args)
    node = LiveVideoProcessor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()