"""Stage 3: learned passing and a collision-checked inside probe.

The opponent runs its own controller. This driver cannot command it to yield.
"""
from maneuver_core import NeuralDriver

class InteractiveModel(NeuralDriver):
    def __init__(self, track, angles, **kwargs):
        super().__init__(track, angles, stage=3, **kwargs)
