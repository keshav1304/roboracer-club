"""Stage 2: learned LEFT / RIGHT / FOLLOW with scan-to-scan motion tracking."""
from maneuver_core import NeuralDriver

class MovingModel(NeuralDriver):
    def __init__(self, track, angles, **kwargs):
        super().__init__(track, angles, stage=2, **kwargs)

