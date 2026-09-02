import rclpy
from rclpy.node import Node
import csv
import os
import yaml
import cv2
import numpy as np
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Pose
from std_msgs.msg import ColorRGBA
from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped

class WaypointVisualizer(Node):
    def __init__(self):
        super().__init__('waypoint_visualizer')
        
        # Publishers
        self.marker_pub = self.create_publisher(Marker, '/autodrive/roboracer_1/waypoint_marker', 10)
        self.path_pub = self.create_publisher(Path, '/autodrive/roboracer_1/path', 10)
        self.map_pub = self.create_publisher(OccupancyGrid, '/autodrive/roboracer_1/map', 10)
        
        # Paths
        pkg_dir = os.path.dirname(__file__)
        self.waypoint_file = os.path.join(pkg_dir, '..', 'comp_waypoints.csv')
        self.map_yaml_file = os.path.join(pkg_dir, '..', 'comp_track.yaml')
        
        # Fallback for user's specific environment
        if not os.path.exists(self.waypoint_file):
            self.waypoint_file = '/home/autodrive_devkit/src/autodrive_devkit/comp_waypoints.csv'
            self.map_yaml_file = '/home/autodrive_devkit/src/autodrive_devkit/comp_track.yaml'

        # Load waypoints
        self.waypoints = []
        try:
            with open(self.waypoint_file, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row:
                        continue
                    try:
                        self.waypoints.append([float(x) for x in row])
                    except ValueError:
                        continue  # skip header / comments
            self.waypoints = np.array(self.waypoints)
            self.get_logger().info(f"Loaded {len(self.waypoints)} waypoints.")
        except Exception as e:
            self.get_logger().error(f"Failed to load waypoints: {e}")

        # Load and Prepare Map
        self.map_msg = self.load_map()

        # Timer to publish visualization periodically
        self.timer = self.create_timer(1.0, self.publish_visualization)

    def load_map(self):
        try:
            with open(self.map_yaml_file, 'r') as f:
                map_meta = yaml.safe_load(f)
            
            resolution = map_meta['resolution']
            origin = map_meta['origin'] # [x, y, yaw]
            pgm_filename = map_meta['image']
            pgm_path = os.path.join(os.path.dirname(self.map_yaml_file), pgm_filename)
            
            img = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise Exception(f"Could not load image {pgm_path}")
            
            height, width = img.shape
            
            grid = OccupancyGrid()
            grid.header.frame_id = "world"
            grid.header.stamp = self.get_clock().now().to_msg()
            grid.info.resolution = resolution
            grid.info.width = width
            grid.info.height = height
            grid.info.origin.position.x = float(origin[0])
            grid.info.origin.position.y = float(origin[1])
            grid.info.origin.position.z = 0.0
            # Simplify orientation for 2D map
            grid.info.origin.orientation.w = 1.0 
            
            # Map PGM (0-255) to OccupancyGrid (0-100)
            # ROS convention: 100 = occupied, 0 = free, -1 = unknown
            # In PGM: 0 = black (occupied), 255 = white (free)
            data = np.zeros((height, width), dtype=np.int8)
            
            # Flip image vertically because OccupancyGrid is bottom-up
            img = np.flipud(img)
            
            occupied_thresh = map_meta.get('occupied_thresh', 0.65)
            free_thresh = map_meta.get('free_thresh', 0.196)
            
            for r in range(height):
                for c in range(width):
                    val = img[r, c]
                    occ = (255.0 - val) / 255.0
                    if occ > occupied_thresh:
                        data[r, c] = 100
                    elif occ < free_thresh:
                        data[r, c] = 0
                    else:
                        data[r, c] = -1
            
            grid.data = data.flatten().tolist()
            self.get_logger().info(f"Loaded map {width}x{height} at {resolution} m/px")
            return grid
        except Exception as e:
            self.get_logger().error(f"Failed to load map: {e}")
            return None

    def publish_visualization(self):
        stamp = self.get_clock().now().to_msg()
        
        # 1. Publish Map
        if self.map_msg:
            self.map_msg.header.stamp = stamp
            self.map_pub.publish(self.map_msg)

        # 2. Publish Waypoints
        if len(self.waypoints) > 0:
            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = stamp
            marker.ns = "waypoints"
            marker.id = 0
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.1
            # Use per-point colors (marker.color is ignored when colors[] is set)
            marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
            
            path = Path()
            path.header.frame_id = "world"
            path.header.stamp = stamp
            
            for wp in self.waypoints:
                p = Point()
                p.x = wp[1]
                p.y = wp[2]
                p.z = 0.05
                marker.points.append(p)
                
                # Color by throttle: red (braking) → dark (coast) → green (full power)
                tv = float(wp[4]) if len(wp) > 4 else 1.0
                tv = max(-1.0, min(1.0, tv))
                c = ColorRGBA()
                if tv >= 0:
                    c.r = 1.0 - tv   # fades from 1→0 as throttle goes 0→1
                    c.g = tv          # fades from 0→1
                    c.b = 0.0
                else:
                    c.r = 1.0           # full red for braking
                    c.g = 0.0
                    c.b = abs(tv) * 0.5  # hint of blue for hard braking
                c.a = 1.0
                marker.colors.append(c)
                
                pose = PoseStamped()
                pose.header = path.header
                pose.pose.position = p
                path.poses.append(pose)
            
            # Close the loop
            if len(marker.points) > 0:
                marker.points.append(marker.points[0])
                marker.colors.append(marker.colors[0])

            self.marker_pub.publish(marker)
            self.path_pub.publish(path)

def main(args=None):
    rclpy.init(args=args)
    visualizer = WaypointVisualizer()
    rclpy.spin(visualizer)
    visualizer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
