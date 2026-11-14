"""Shared local collision geometry for reactive LiDAR controllers.

All geometry is rebuilt in the car's local frame from the current scan.
There is no saved map, racing line, checkpoint, pose or other-car state.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates


class LidarPreview:
    """Check forward steering sequences toward a local reactive objective."""

    def path_cost(self, x, y, yaw, field, resolution, radius):
        return np.zeros(x.shape[:2])

    def __init__(self):
        self.controls = np.array([-.4189, -.32, -.21, -.1, 0., .1, .21, .32, .4189])
        self.curvatures = np.tan(self.controls) / .3302
        self.segment_length = .5
        self.sample_spacing = .1
        self.depth = 8  # four metres, enough to see both sides of a hairpin
        samples = np.arange(1, 6) * self.sample_spacing
        self.headings = self.curvatures[:, None] * samples
        denominator = np.where(abs(self.curvatures) > .0001, self.curvatures, 1.)
        self.local_x = np.sin(self.headings) / denominator[:, None]
        self.local_y = (1 - np.cos(self.headings)) / denominator[:, None]
        self.local_x[abs(self.curvatures) < .0001] = samples
        self.last_steering = 0.
        self.lateral_acceleration = 2.5
        self.steering_weight = .04
        self.clearance_reward_cap = .75
        self.resolution = .025
        self.body_offsets = np.array([-.2175, -.0725, .0725, .2175])
        self.body_radius = .205

    def plan(self, scan, angles, goal_heading, desired_steering, v_max):
        resolution, radius = self.resolution, 6.
        grid_size = round(2 * radius / resolution) + 1
        grid = np.ones((grid_size, grid_size), bool)
        points = scan[:, None] * np.column_stack((np.cos(angles), np.sin(angles)))
        cells = np.rint((points + radius) / resolution).astype(int)
        inside = (cells >= 0).all(axis=1) & (cells < grid_size).all(axis=1)
        cells = cells[inside]
        grid[cells[:, 1], cells[:, 0]] = False
        field = distance_transform_edt(grid) * resolution

        # x, y, heading, first control index, last steering, accumulated merit.
        nodes = np.zeros((1, 6))
        nodes[0, 3] = -1
        nodes[0, 4] = self.last_steering
        paths = np.zeros((1, 0, 4))
        target = np.array([np.cos(goal_heading), np.sin(goal_heading)])
        chosen = None
        for depth in range(self.depth):
            heading = nodes[:, 2, None, None]
            x = (nodes[:, 0, None, None] + np.cos(heading) * self.local_x[None]
                 - np.sin(heading) * self.local_y[None])
            y = (nodes[:, 1, None, None] + np.sin(heading) * self.local_x[None]
                 + np.cos(heading) * self.local_y[None])
            yaw = heading + self.headings[None]
            centers = np.stack((x, y), axis=-1)
            tangent = np.stack((np.cos(yaw), np.sin(yaw)), axis=-1)
            # The controller's overlapping-circle profile encloses the
            # 0.58 x 0.31 m body with room for scan and grid uncertainty.
            body = centers[..., None, :] + tangent[..., None, :] * self.body_offsets[None, None, None, :, None]
            coordinates = (body.reshape(-1, 2) + radius) / resolution
            distances = map_coordinates(
                field, [coordinates[:, 1], coordinates[:, 0]],
                order=1, mode="constant", cval=0.
            ).reshape(*x.shape, len(self.body_offsets)).min(axis=-1)
            # A gap hidden behind a wall or another car is not confirmed
            # free space. Prefer scan-visible trajectories without requiring
            # a saved occupancy map or vehicle pose history.
            radial = np.hypot(x, y)
            visible_range = np.interp(
                np.arctan2(y, x).ravel(), angles, scan, left=0., right=0.
            ).reshape(x.shape)
            known = radial < visible_range - .08
            valid = (distances.min(axis=-1) > self.body_radius) & (abs(yaw[:, :, -1]) < 2.7)
            if not valid.any():
                break

            path_cost = self.path_cost(x, y, yaw, field, resolution, radius)
            parent, control = np.where(valid)
            first = np.where(depth == 0, control, nodes[parent, 3]).astype(int)
            steering = self.controls[control]
            merit = (nodes[parent, 5] + .6 * np.minimum(distances[parent, control], self.clearance_reward_cap).mean(axis=1)
                     - .7 * abs(steering - nodes[parent, 4])
                     - .6 * (1 - known[parent, control].mean(axis=1))
                     - path_cost[parent, control])
            endpoints = np.column_stack((x[parent, control, -1], y[parent, control, -1]))
            rank = (merit + .18 * (endpoints @ target)
                    - self.steering_weight * abs(self.controls[first] - desired_steering))
            expanded = np.column_stack((endpoints, yaw[parent, control, -1], first, steering, merit))
            steps = np.stack((x[parent, control], y[parent, control], yaw[parent, control],
                              np.repeat(self.curvatures[control, None], 5, axis=1)), axis=-1)
            expanded_paths = np.concatenate((paths[parent], steps), axis=1)
            # Keep alternatives for every initial steering choice, so a
            # briefly narrower entrance cannot hide a clear bend beyond it.
            keep = []
            for initial in np.unique(first):
                indices = np.flatnonzero(first == initial)
                keep.extend(indices[np.argsort(rank[indices])[-4:]])
            keep = np.asarray(keep, int)
            nodes, paths = expanded[keep], expanded_paths[keep]
            best = int(np.argmax(rank[keep]))
            chosen = nodes[best], paths[best], depth + 1

        if chosen is None:
            return float(desired_steering), 0.
        node, path, depth = chosen
        steering = float(self.controls[int(node[3])])
        along = np.arange(1, len(path) + 1) * self.sample_spacing
        corner_speed = np.sqrt(self.lateral_acceleration / np.maximum(abs(path[:, 3]), .01))
        # Brake before a bend in the local preview, rather than waiting for
        # the commanded steering angle to become large.
        allowed = np.sqrt(corner_speed ** 2 + 6 * np.maximum(along - .25, 0.))
        speed = min(v_max, float(allowed.min()),
                    np.sqrt(max(6 * (depth * self.segment_length - .45), 0.)))
        self.last_steering = steering
        return steering, float(speed)
