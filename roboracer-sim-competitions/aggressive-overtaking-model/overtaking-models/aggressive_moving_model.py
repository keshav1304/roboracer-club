"""Stage 4 configuration: frozen aggressive Stage 3 policy with speed advantage.

This reuses trained weights; it is not a newly trained network. The independent
opponent is the conservative Stage 2 policy at a lower configured speed cap.
"""
from interactive_model import InteractiveModel

class AggressiveMovingModel(InteractiveModel):
    pass
