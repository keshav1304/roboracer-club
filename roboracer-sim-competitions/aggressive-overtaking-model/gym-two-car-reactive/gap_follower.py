"""LiDAR follow-the-gap with a short, dynamically checked corner preview."""
import numpy as np
from reactive import FollowTheGap, FTGParams
from gap_preview import GapPreview


class GapFollower(FollowTheGap):
    def __init__(self, angles, v_max=3.6, safety_margin=.15):
        super().__init__(angles, FTGParams(
            v_max=v_max, v_min=.6, fov_deg=170, max_range=8.,
            lookahead_min=.7, lookahead_max=1.5, margin=safety_margin,
            bubble=.25, deep_frac=.9, side_lo_deg=65., side_safe=.4,
            side_gain=0., clear_ok=0., stop_dist=.4, corridor_margin=.08,
            reverse_time=0.
        ))
        self.preview = GapPreview()
        self.side_sector = (np.abs(self.angles) >= np.deg2rad(65)) & (np.abs(self.angles) <= np.deg2rad(110))
        self.proximity_speed_cap = float("inf")
        self.tick = 0
        self.command = (0., 0.)

    def plan(self, scan, speed=0.0):
        scan = np.clip(np.nan_to_num(np.asarray(scan, float), nan=10., posinf=10., neginf=0.), 0, 30)
        side = float(np.percentile(scan[self.side_sector], 5))
        # Reduce speed early when lateral room closes, while a moving
        # obstacle still has room to clear the car's front and rear corners.
        self.proximity_speed_cap = (float("inf") if side >= .65 else
                                    max(.4, 1.6 * (side - .25) / .4))
        if np.max(scan) < .35:
            self.tick = 0
            self.command = (0., 0.)
            return self.command
        # Plan at 10 Hz. The shared safety layer checks current scan clearance
        # and applies steering/speed limits on every 100 Hz physics step.
        if self.tick % 10 == 0:
            gap_steering, _ = super().plan(scan, speed)
            self.command = self.preview.plan(
                scan, self.angles, self.last_target_angle, gap_steering, self.p.v_max
            )
        self.tick += 1
        return self.command[0], min(self.command[1], self.proximity_speed_cap)
