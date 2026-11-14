"""LiDAR collision preview with a left-wall distance objective."""
import numpy as np
from scipy.ndimage import map_coordinates
from lidar_preview import LidarPreview


class WallPreview(LidarPreview):
    def __init__(self, desired_distance):
        super().__init__()
        self.desired_distance = desired_distance
        self.lateral_acceleration = 1.5
        self.steering_weight = .2
        self.clearance_reward_cap = .45

    def path_cost(self, x, y, yaw, field, resolution, radius):
        # On a well-aligned wall-following path, points this far to the left
        # lie on the observed wall. This is a wall-standoff objective,
        # independent of any free-gap target or saved track geometry.
        left_x = x - self.desired_distance * np.sin(yaw)
        left_y = y + self.desired_distance * np.cos(yaw)
        coordinates = np.stack(((left_y + radius) / resolution,
                                (left_x + radius) / resolution))
        distances = map_coordinates(
            field, coordinates.reshape(2, -1), order=1, mode="constant", cval=1.
        ).reshape(x.shape)
        # Bound the cost when an inside wall ends, allowing a clear turn
        # around the endpoint instead of a late switch to the outer wall.
        return .35 * np.minimum(distances, 1.).mean(axis=-1)

    def plan(self, scan, angles, wall_steering, v_max):
        heading = float(np.clip(2 * wall_steering, -1., 1.))
        return super().plan(scan, angles, heading, wall_steering, v_max)
