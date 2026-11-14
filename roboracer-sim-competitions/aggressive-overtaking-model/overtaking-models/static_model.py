"""Stage 1: learned LEFT / RIGHT / WAIT for a LiDAR-detected static obstacle."""
from maneuver_core import NeuralDriver

class StaticModel(NeuralDriver):
    def __init__(self, track, angles, **kwargs):
        super().__init__(track, angles, stage=1, **kwargs)

