"""LiDAR turn preview for the opening selected by follow-the-gap."""
import numpy as np
from lidar_preview import LidarPreview


class GapPreview(LidarPreview):
    def __init__(self):
        super().__init__()
        # The faster car keeps a wider footprint cushion and looks farther
        # through a corner before committing to an opening.
        self.depth = 10
        self.resolution = .05
        self.body_offsets = np.array([-.145, .145])
        self.body_radius = .25
