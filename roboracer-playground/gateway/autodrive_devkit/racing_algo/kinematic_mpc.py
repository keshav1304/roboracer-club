# MIT License
#
# Copyright (c) Hongrui Zheng, Johannes Betz
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Kinematic bicycle MPC waypoint tracker.

Adapted from f1tenth_planning KMPCPlanner (CVXPY / OSQP). Pose and
waypoints are in the map frame. Waypoints are passed as
[cx, cy, cyaw, speed] column lists.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cvxpy
import numpy as np
from scipy.linalg import block_diag as np_block_diag
from scipy.sparse import block_diag, csc_matrix


def nearest_point(point, trajectory):
    """Nearest point on a piecewise-linear trajectory (no numba)."""
    diffs = trajectory[1:, :] - trajectory[:-1, :]
    l2s = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
    l2s = np.maximum(l2s, 1e-12)
    dots = np.sum((point - trajectory[:-1, :]) * diffs, axis=1)
    t = np.clip(dots / l2s, 0.0, 1.0)
    projections = trajectory[:-1, :] + (t * diffs.T).T
    dists = np.linalg.norm(point - projections, axis=1)
    i = int(np.argmin(dists))
    return projections[i], float(dists[i]), float(t[i]), i


@dataclass
class MpcConfig:
    NXK: int = 4
    NU: int = 2
    TK: int = 8
    Rk: np.ndarray = field(
        default_factory=lambda: np.diag([0.01, 100.0]))
    Rdk: np.ndarray = field(
        default_factory=lambda: np.diag([0.01, 100.0]))
    Qk: np.ndarray = field(
        default_factory=lambda: np.diag([13.5, 13.5, 5.5, 13.0]))
    Qfk: np.ndarray = field(
        default_factory=lambda: np.diag([13.5, 13.5, 5.5, 13.0]))
    DTK: float = 0.1
    dlk: float = 0.03
    WB: float = 0.33
    MIN_STEER: float = -0.4189
    MAX_STEER: float = 0.4189
    MAX_DSTEER: float = np.deg2rad(180.0)
    MAX_SPEED: float = 6.0
    MIN_SPEED: float = 0.0
    MAX_ACCEL: float = 3.0


@dataclass
class State:
    x: float = 0.0
    y: float = 0.0
    delta: float = 0.0
    v: float = 0.0
    yaw: float = 0.0
    yawrate: float = 0.0
    beta: float = 0.0


class KMPCPlanner:
    """Linearized kinematic bicycle MPC over a finite horizon."""

    def __init__(self, waypoints=None, config: MpcConfig | None = None):
        self.waypoints = waypoints
        self.config = config or MpcConfig()
        self.oa = None
        self.odelta_v = None
        self.last_pred_x = np.zeros(0)
        self.last_pred_y = np.zeros(0)
        self.last_ref_x = np.zeros(0)
        self.last_ref_y = np.zeros(0)
        self.mpc_prob_init_kinematic()

    def plan(self, states):
        """
        Args:
            states: [x, y, delta, v, yaw, yawrate, beta]
        Returns:
            steering_angle (rad), speed (m/s). On solver failure returns
            (None, None).
        """
        if self.waypoints is None:
            raise ValueError("waypoints not set")
        vehicle_state = State(
            x=float(states[0]),
            y=float(states[1]),
            delta=float(states[2]),
            v=float(states[3]),
            yaw=float(states[4]),
            yawrate=float(states[5]),
            beta=float(states[6]),
        )
        speed, steer = self._mpc_control(vehicle_state, self.waypoints)
        return steer, speed

    def calc_ref_trajectory(self, state, cx, cy, cyaw, sp):
        ref_traj = np.zeros((self.config.NXK, self.config.TK + 1))
        ncourse = len(cx)
        _, _, _, ind = nearest_point(
            np.array([state.x, state.y]), np.array([cx, cy]).T)

        travel = abs(state.v) * self.config.DTK
        dind = max(travel / self.config.dlk, 1e-3)
        ind_list = int(ind) + np.insert(
            np.cumsum(np.repeat(dind, self.config.TK)), 0, 0
        ).astype(int)
        ind_list = np.mod(ind_list, ncourse)

        cyaw = np.array(cyaw, dtype=float, copy=True)
        # Unwrap heading jumps relative to the vehicle yaw for the QP.
        for i in range(len(cyaw)):
            d = cyaw[i] - state.yaw
            while d > math.pi:
                cyaw[i] -= 2.0 * math.pi
                d -= 2.0 * math.pi
            while d < -math.pi:
                cyaw[i] += 2.0 * math.pi
                d += 2.0 * math.pi

        ref_traj[0, :] = cx[ind_list]
        ref_traj[1, :] = cy[ind_list]
        ref_traj[2, :] = sp[ind_list]
        ref_traj[3, :] = cyaw[ind_list]
        return ref_traj

    def predict_motion(self, x0, oa, od):
        path_predict = np.zeros((self.config.NXK, self.config.TK + 1))
        for i, _ in enumerate(x0):
            path_predict[i, 0] = x0[i]
        state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
        for ai, di, i in zip(oa, od, range(1, self.config.TK + 1)):
            state = self.update_state(state, ai, di)
            path_predict[0, i] = state.x
            path_predict[1, i] = state.y
            path_predict[2, i] = state.v
            path_predict[3, i] = state.yaw
        return path_predict

    def update_state(self, state, a, delta):
        delta = max(self.config.MIN_STEER, min(self.config.MAX_STEER, delta))
        state.x = state.x + state.v * math.cos(state.yaw) * self.config.DTK
        state.y = state.y + state.v * math.sin(state.yaw) * self.config.DTK
        state.yaw = (
            state.yaw
            + (state.v / self.config.WB) * math.tan(delta) * self.config.DTK
        )
        state.v = state.v + a * self.config.DTK
        state.v = max(
            self.config.MIN_SPEED, min(self.config.MAX_SPEED, state.v))
        return state

    def get_kinematic_model_matrix(self, v, phi, delta):
        A = np.zeros((self.config.NXK, self.config.NXK))
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 2] = 1.0
        A[3, 3] = 1.0
        A[0, 2] = self.config.DTK * math.cos(phi)
        A[0, 3] = -self.config.DTK * v * math.sin(phi)
        A[1, 2] = self.config.DTK * math.sin(phi)
        A[1, 3] = self.config.DTK * v * math.cos(phi)
        A[3, 2] = self.config.DTK * math.tan(delta) / self.config.WB

        B = np.zeros((self.config.NXK, self.config.NU))
        B[2, 0] = self.config.DTK
        cos_d = math.cos(delta)
        B[3, 1] = self.config.DTK * v / (self.config.WB * cos_d ** 2)

        C = np.zeros(self.config.NXK)
        C[0] = self.config.DTK * v * math.sin(phi) * phi
        C[1] = -self.config.DTK * v * math.cos(phi) * phi
        C[3] = -self.config.DTK * v * delta / (self.config.WB * cos_d ** 2)
        return A, B, C

    @staticmethod
    def _flatten(x):
        return np.array(x).flatten()

    def mpc_prob_init_kinematic(self):
        cfg = self.config
        self.xk = cvxpy.Variable((cfg.NXK, cfg.TK + 1))
        self.uk = cvxpy.Variable((cfg.NU, cfg.TK))
        objective = 0.0
        constraints = []

        self.x0k = cvxpy.Parameter(cfg.NXK)
        self.x0k.value = np.zeros(cfg.NXK)
        self.ref_traj_k = cvxpy.Parameter((cfg.NXK, cfg.TK + 1))
        self.ref_traj_k.value = np.zeros((cfg.NXK, cfg.TK + 1))

        R_block = np_block_diag(*([cfg.Rk] * cfg.TK))
        Rd_block = np_block_diag(*([cfg.Rdk] * (cfg.TK - 1)))
        Q_list = [cfg.Qk] * cfg.TK + [cfg.Qfk]
        Q_block = np_block_diag(*Q_list)

        objective += cvxpy.quad_form(cvxpy.vec(self.uk), R_block)
        objective += cvxpy.quad_form(
            cvxpy.vec(self.xk - self.ref_traj_k), Q_block)
        objective += cvxpy.quad_form(
            cvxpy.vec(cvxpy.diff(self.uk, axis=1)), Rd_block)

        path_predict = np.zeros((cfg.NXK, cfg.TK + 1))
        A_block = []
        B_block = []
        C_block = []
        for t in range(cfg.TK):
            A, B, C = self.get_kinematic_model_matrix(
                path_predict[2, t], path_predict[3, t], 0.0)
            A_block.append(A)
            B_block.append(B)
            C_block.extend(C)
        A_block = block_diag(tuple(A_block))
        B_block = block_diag(tuple(B_block))
        C_block = np.array(C_block)

        m, n = A_block.shape
        self.Annz_k = cvxpy.Parameter(A_block.nnz)
        data = np.ones(self.Annz_k.size)
        rows = A_block.row * n + A_block.col
        cols = np.arange(self.Annz_k.size)
        indexer_a = csc_matrix(
            (data, (rows, cols)), shape=(m * n, self.Annz_k.size))
        self.Annz_k.value = A_block.data
        self.Ak_ = cvxpy.reshape(
            indexer_a @ self.Annz_k, (m, n), order="C")

        m, n = B_block.shape
        self.Bnnz_k = cvxpy.Parameter(B_block.nnz)
        data = np.ones(self.Bnnz_k.size)
        rows = B_block.row * n + B_block.col
        cols = np.arange(self.Bnnz_k.size)
        indexer_b = csc_matrix(
            (data, (rows, cols)), shape=(m * n, self.Bnnz_k.size))
        self.Bk_ = cvxpy.reshape(
            indexer_b @ self.Bnnz_k, (m, n), order="C")
        self.Bnnz_k.value = B_block.data

        self.Ck_ = cvxpy.Parameter(C_block.shape)
        self.Ck_.value = C_block

        constraints += [
            cvxpy.vec(self.xk[:, 1:])
            == self.Ak_ @ cvxpy.vec(self.xk[:, :-1])
            + self.Bk_ @ cvxpy.vec(self.uk)
            + self.Ck_
        ]
        constraints += [
            cvxpy.abs(cvxpy.diff(self.uk[1, :]))
            <= cfg.MAX_DSTEER * cfg.DTK
        ]
        constraints += [self.xk[:, 0] == self.x0k]
        constraints += [self.xk[2, :] <= cfg.MAX_SPEED]
        constraints += [self.xk[2, :] >= cfg.MIN_SPEED]
        constraints += [cvxpy.abs(self.uk[0, :]) <= cfg.MAX_ACCEL]
        constraints += [cvxpy.abs(self.uk[1, :]) <= cfg.MAX_STEER]

        self.MPC_prob = cvxpy.Problem(cvxpy.Minimize(objective), constraints)

    def mpc_prob_solve(self, ref_traj, path_predict, x0):
        cfg = self.config
        self.x0k.value = x0

        A_block = []
        B_block = []
        C_block = []
        for t in range(cfg.TK):
            A, B, C = self.get_kinematic_model_matrix(
                path_predict[2, t], path_predict[3, t], 0.0)
            A_block.append(A)
            B_block.append(B)
            C_block.extend(C)
        A_block = block_diag(tuple(A_block))
        B_block = block_diag(tuple(B_block))
        C_block = np.array(C_block)

        self.Annz_k.value = A_block.data
        self.Bnnz_k.value = B_block.data
        self.Ck_.value = C_block
        self.ref_traj_k.value = ref_traj

        self.MPC_prob.solve(solver=cvxpy.OSQP, verbose=False, warm_start=True)

        if self.MPC_prob.status in (
                cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE):
            ox = self._flatten(self.xk.value[0, :])
            oy = self._flatten(self.xk.value[1, :])
            oa = self._flatten(self.uk.value[0, :])
            odelta = self._flatten(self.uk.value[1, :])
            return oa, odelta, ox, oy
        return None, None, None, None

    def _mpc_control(self, vehicle_state, path):
        cx, cy, cyaw, sp = path[0], path[1], path[2], path[3]
        ref_path = self.calc_ref_trajectory(
            vehicle_state, cx, cy, cyaw, sp)
        x0 = [
            vehicle_state.x, vehicle_state.y,
            vehicle_state.v, vehicle_state.yaw,
        ]

        oa = self.oa
        od = self.odelta_v
        if oa is None or od is None:
            oa = [0.0] * self.config.TK
            od = [0.0] * self.config.TK

        path_predict = self.predict_motion(x0, oa, od)
        mpc_a, mpc_delta, ox, oy = self.mpc_prob_solve(
            ref_path, path_predict, x0)

        self.last_ref_x = np.array(ref_path[0], dtype=float)
        self.last_ref_y = np.array(ref_path[1], dtype=float)

        if mpc_a is None or mpc_delta is None:
            self.last_pred_x = np.array(path_predict[0], dtype=float)
            self.last_pred_y = np.array(path_predict[1], dtype=float)
            return None, None

        self.oa = mpc_a
        self.odelta_v = mpc_delta
        self.last_pred_x = np.array(ox, dtype=float)
        self.last_pred_y = np.array(oy, dtype=float)

        steer_output = float(mpc_delta[0])
        speed_output = float(
            vehicle_state.v + mpc_a[0] * self.config.DTK)
        speed_output = max(
            self.config.MIN_SPEED,
            min(self.config.MAX_SPEED, speed_output))
        return speed_output, steer_output
