"""Vehicle physics for env_v2. Two models of the same AutoDRIVE car, trained on together.

MODEL_TWIN, a physics model built from AutoDRIVE's own source, with the free numbers fitted to our logs.
MODEL_EMPIRICAL, the KS + slide model of ../env/race_env.py (the one P5 was validated on), ported as is.
A policy has to drive both, so it cannot exploit a mistake that only one of them makes.

The twin, in one paragraph. AutoDRIVE's F1TENTH is a Unity car with four WheelColliders. Every tyre force
is (friction curve value at the current slip) x (curve stiffness) x (load on the wheel), and the
`TireFriction` script scales the stiffness of the forward AND the sideways curve by one friction
coefficient. So one number, MU here, sets how hard the car can accelerate, brake and turn, and when it
slides. The rest follows from VehicleController.cs and the Sim Racing scene.

Lateral. Per axle, F_y = MU * SY * g(wheel) * N_axle * curve_y(slip angle). curve_y is Unity's two-piece
spline, extremum (0.01, 1.0) and asymptote (0.1, 0.5) from the prefab. It peaks at a tiny slip and then
falls to half, which is why the AutoDRIVE car holds ~10 m/s^2 and then suddenly slides at ~4.5 m/s^2.
g(wheel) = 1 up to 14 deg and falls to ~0.75 at 18 deg (fitted to the logs, see AUTODRIVE_PHYSICS.md 1).
Longitudinal. driveType CAWD (all four wheels driven) and the wheel damping 0.25 make the wheels run at
v* = MotorTorque/4 * r / damping * throttle = 428/4 * 0.059 / 0.25 * throttle = 25.25 * throttle.
The tyres push the car toward v* (K_SERVO), friction-limited:
    throttle > 0   a = clip(K_SERVO * (v* - v), -MU * A_EB, +MU * A_DRIVE)
    throttle == 0  a = -MU * A_BRAKE   (VehicleController applies brakeTorque = MotorTorque to all wheels)
plus Rigidbody drag 0.273 * v (prefab m_Drag). A_DRIVE, A_EB, A_BRAKE are fitted so the totals match the
logged 4.7 m/s^2 ramp, 4.3 to 4.5 m/s^2 engine braking and 5.7 m/s^2 at throttle 0 (8 -> 4 m/s in 4.2 m).
Coupling. A tyre that is driving or braking may have less sideways grip. F_y *= sqrt(1 - K_COUPLE * u^2)
with u = |F_x| / (MU * XPEAK * g * N). Unity's WheelCollider probably keeps the two directions separate
(nominal 0), randomised over [0, 0.5] until the probe node measures it.
Load transfer. N_front = m (g l_r - a_x h) / L, so braking loads the front and unloads the rear.
Balance. Every slide in our logs was the front running wide, also while braking in corners (the grip shield
coasts = brakes all four wheels at 9.2 m/s^2 of demand), and AutoDRIVE holds 10.4 m/s^2 for a moment but
slides when ~9.5 is held for half a second. Three numbers reproduce all of that together:
  FRONT_PEAK 0.95  the two Ackermann-steered front wheels never reach their peak slip together,
  REAR_GRIP 1.25   so the rear keeps a margin, also when braking moves load forward,
  IZ 0.10          yaw inertia, which is what makes a short spike of demand survivable and a long one not.
Searched on a grid against validate_v2.py: PP x1.10/x1.15 clean, x1.20 slides at the front, P5 clean, and
the whole neighbourhood iz 0.07-0.14, front 0.95, rear 1.2-1.35 also matches. The probe node checks them.
Steering. VehicleController.Steer() ported line by line. The angle grows at 183.346 deg/s, but whenever
the new command is SMALLER in magnitude than the current angle it jumps there in one physics step (the
comparison in Steer() mixes the sign conventions). Plus the command delay measured in our logs (~30 ms).
Understeer. The front sees atan(K_US tan(wheel)), K_US 0.955 from the logs (AUTODRIVE_PHYSICS.md 3).

Integration is semi-implicit Euler at 0.25 ms (the tyres are stiff), with a blend to the kinematic bicycle
below 2 m/s like the gym's ST model.
"""
from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit
except ImportError:                                   # works without numba, only slower
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

from contract import CAR_LENGTH, CAR_WIDTH, MASS, MAX_STEER, STEER_RATE_DEG, WHEELBASE

G = 9.81
MODEL_TWIN, MODEL_EMPIRICAL = 0, 1
DT_PHYS = 0.00025             # s
COLLISION_EVERY = 40          # physics steps between wall checks (10 ms)
SLIDE_SLIP = 4.0              # a tyre counts as sliding past 4 x its extremum slip (force already falling)

# ---- parameter vector -----------------------------------------------------------------------------
NAMES = ['model', 'mu', 'sy', 'ext_s', 'ext_v', 'asy_s', 'asy_v', 'g_lo', 'd1', 'd2', 'k_us', 'k_couple',
         'xpeak', 'a_drive', 'a_eb', 'a_brake', 'k_servo', 'v_per_thr', 'drag', 'ang_drag', 'm', 'iz', 'lf',
         'lr', 'h', 'steer_rate', 'delay_s', 'delay_t', 'max_steer',
         # empirical model only (../env/race_env.py names)
         'e_a_hi', 'e_a_lo', 'e_d1', 'e_d2', 'e_slide_trig', 'e_a_slide', 'e_turn_drag', 'e_a_up', 'e_coast',
         'e_coast_zero', 'e_k_speed', 'e_v_per_thr',
         'rear_grip', 'front_peak']
IDX = {n: i for i, n in enumerate(NAMES)}
NP = len(NAMES)

NOMINAL = dict(
    model=MODEL_TWIN,
    mu=1.0,                       # TireFriction.frictionCoefficient (1.0 = the scene's own curves)
    sy=1.0,                       # sideways stiffness, prefab
    ext_s=0.01, ext_v=1.0, asy_s=0.1, asy_v=0.5,      # sideways curve, prefab
    g_lo=0.75, d1=math.radians(14.0), d2=math.radians(18.0),   # front grip by wheel angle, fitted to logs
    k_us=0.955,                   # logs
    k_couple=0.0,                 # unknown. Braking in corners (the grip shield) never spun the car, so nominal 0
    xpeak=0.72,                   # forward curve extremum 0.9 x stiffness 0.8, prefab
    a_drive=5.4, a_eb=2.9, a_brake=4.1,   # m/s^2 at mu 1, fitted so totals with drag match the logs
    k_servo=8.0,                  # 1/s, logs (how sharply speed settles onto v*)
    v_per_thr=25.25,              # 428 / 4 * 0.059 / 0.25, Sim Racing scene + prefab
    drag=0.273, ang_drag=0.1,     # Rigidbody m_Drag, m_AngularDrag
    m=MASS, iz=0.10,              # Unity derives it from the colliders (prefab tensor is a placeholder); 0.07 to 0.14
                                  # all reproduce AutoDRIVE, 0.044 (the gym's, scaled) does not
    lf=0.1747, lr=0.1553,         # prefab: wheel colliders at +0.17 / -0.16 m, COM 4.7 mm behind the origin
    h=0.065,                      # m, prefab COM y 0.064 with the body origin at ground level (wheel centre 0.06, r 0.059)
    steer_rate=math.radians(STEER_RATE_DEG),
    delay_s=0.030, delay_t=0.040,  # s, logs
    max_steer=MAX_STEER,
    e_a_hi=10.4, e_a_lo=7.0, e_d1=math.radians(14.0), e_d2=math.radians(18.0), e_slide_trig=0.08,
    e_a_slide=4.5, e_turn_drag=3.0, e_a_up=4.7, e_coast=4.5, e_coast_zero=5.7, e_k_speed=8.0,
    e_v_per_thr=24.6,
    rear_grip=1.25,               # rear sideways grip relative to the scene's curve (see Balance)
    front_peak=0.95,               # front axle peak relative to the curve (see Balance)
)

# per-episode randomisation: name -> (low, high). Wider than what we believe on purpose.
DR_RANGES = dict(
    # friction: about the calibration's own uncertainty. P5 and PP x1.15 sit ~9 % of demand under the slide,
    # so wider bands only teach "always slower", which AutoDRIVE does not need (the probe node can widen them)
    mu=(0.97, 1.03), sy=(0.99, 1.01), ext_s=(0.009, 0.012), asy_v=(0.45, 0.55), g_lo=(0.72, 0.78),
    d1=(math.radians(13.5), math.radians(14.5)), k_us=(0.945, 0.965), k_couple=(0.0, 0.2), rear_grip=(1.2, 1.35),
    front_peak=(0.94, 0.97), iz=(0.08, 0.13), h=(0.055, 0.075),
    # drivetrain
    a_drive=(5.1, 5.7), a_eb=(2.7, 3.1), a_brake=(3.9, 4.3), k_servo=(7.0, 9.0), v_per_thr=(24.9, 25.6),
    drag=(0.26, 0.285), m=(3.40, 3.55),
    # lag: wide on purpose, this is what broke policies in AutoDRIVE
    steer_rate=(math.radians(170.0), math.radians(195.0)), delay_s=(0.020, 0.050), delay_t=(0.020, 0.060),
    # the empirical model's own band (../env/race_env.py)
    e_a_hi=(10.1, 10.7), e_a_lo=(7.0, 8.5), e_d1=(math.radians(13.0), math.radians(15.0)),
    e_slide_trig=(0.06, 0.10), e_a_slide=(4.0, 5.5), e_turn_drag=(2.0, 4.0), e_a_up=(4.25, 5.15),
)


def nominal_params(model: int = MODEL_TWIN) -> np.ndarray:
    p = np.array([NOMINAL[n] for n in NAMES], dtype=np.float64)
    p[IDX['model']] = model
    return p


def sample_params(rng: np.random.Generator, model: int = MODEL_TWIN, randomize: bool = True) -> np.ndarray:
    p = nominal_params(model)
    if not randomize:
        return p
    for name, (lo, hi) in DR_RANGES.items():
        p[IDX[name]] = rng.uniform(lo, hi)
    p[IDX['d2']] = p[IDX['d1']] + math.radians(rng.uniform(3.5, 4.5))
    p[IDX['e_d2']] = p[IDX['e_d1']] + math.radians(rng.uniform(3.5, 4.5))
    shift = rng.uniform(-0.006, 0.006)                         # centre of mass a few mm off centre
    p[IDX['lf']], p[IDX['lr']] = NOMINAL['lf'] + shift, NOMINAL['lr'] - shift
    f = rng.uniform(0.9, 1.1)                                  # empirical coasting, as in ../env/race_env.py
    p[IDX['e_coast']] *= f
    p[IDX['e_coast_zero']] *= f
    return p


# ---- state vector -------------------------------------------------------------------------------
S_X, S_Y, S_PSI, S_VX, S_VY, S_R, S_STEER, S_AX, S_T, S_SLIDE, S_OVER = range(11)
NS = 11
# outputs of simulate(): collided, max front slip ratio (|slip| / extremum), max rear, max body slip (rad),
# time sliding front (s), time sliding rear (s), steps run
O_COL, O_SF, O_SR, O_BETA, O_TSF, O_TSR = range(6)


def new_state(x: float, y: float, yaw: float, speed: float = 0.0) -> np.ndarray:
    s = np.zeros(NS)
    s[S_X], s[S_Y], s[S_PSI], s[S_VX] = x, y, yaw, speed
    return s


def wheel_angle(state) -> float:
    """Current physical wheel angle, rad, +left."""
    return -math.radians(state[S_STEER])


def speed_of(state) -> float:
    return math.hypot(state[S_VX], state[S_VY])


@njit(cache=True)
def _curve(s, ext_s, ext_v, asy_s, asy_v):
    """Unity WheelFrictionCurve: rise to the extremum, fall to the asymptote, flat after. s >= 0."""
    if s <= ext_s:
        t = s / ext_s
        return ext_v * t * (2.0 - t)
    if s >= asy_s:
        return asy_v
    t = (s - ext_s) / (asy_s - ext_s)
    return ext_v + (asy_v - ext_v) * t * t * (3.0 - 2.0 * t)


@njit(cache=True)
def _front_grip(w, g_lo, d1, d2):
    w = abs(w)
    if w <= d1:
        return 1.0
    if w >= d2:
        return g_lo
    return 1.0 + (g_lo - 1.0) * (w - d1) / (d2 - d1)


@njit(cache=True)
def _autodrive_steer(S, c, rate_deg_s, limit_deg, dt):
    """VehicleController.Steer(), autonomous branch, line by line. S is Unity's SteeringAngle in deg,
    which is minus our wheel angle. c is the steering_command."""
    if c > S / limit_deg:
        S -= rate_deg_s * dt
        if S <= -c * limit_deg:
            S = -c * limit_deg
    else:
        S += rate_deg_s * dt
        if S >= -c * limit_deg:
            S = -c * limit_deg
    return S


@njit(cache=True)
def _wall_hit(x, y, psi, edt, ox, oy, res, half_l, half_w):
    c, s = math.cos(psi), math.sin(psi)
    H, W = edt.shape
    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:
                continue
            lx = (i - 1) * half_l
            ly = (j - 1) * half_w
            px = x + c * lx - s * ly
            py = y + s * lx + c * ly
            r = int((py - oy) / res)
            q = int((px - ox) / res)
            if r < 0 or q < 0 or r >= H or q >= W:
                return True
            if edt[r, q] <= 0.0:
                return True
    return False


@njit(cache=True)
def _twin_derivs(st, p, wheel, thr):
    mu, sy = p[1], p[2]
    m, iz, lf, lr = p[20], p[21], p[22], p[23]
    L = lf + lr
    vx = max(st[S_VX], 0.0)
    vy, r = st[S_VY], st[S_R]
    # longitudinal request, friction limited
    if thr <= 0.0:
        a_req = -mu * p[15] if vx > 0.05 else -vx / 0.01
    else:
        a_req = p[16] * (p[17] * thr - vx)
        a_req = min(mu * p[13], max(-mu * p[14], a_req))
    ax_prev = st[S_AX]
    nf = max(0.05 * m * G, m * (G * lr - ax_prev * p[24]) / L)
    nr = max(0.05 * m * G, m * (G * lf + ax_prev * p[24]) / L)
    fx = m * a_req
    fxf = fx * nf / (nf + nr)
    fxr = fx - fxf
    # slip angles
    vxs = max(vx, 0.1)
    w_eff = math.atan(p[10] * math.tan(wheel))
    af = w_eff - math.atan2(vy + lf * r, vxs)
    ar = -math.atan2(vy - lr * r, vxs)
    gf = _front_grip(wheel, p[7], p[8], p[9])                      # lowers the front's peak, not its slide level
    uf = min(1.0, abs(fxf) / max(1e-6, mu * p[12] * nf))          # share of the forward grip in use
    ur = min(1.0, abs(fxr) / max(1e-6, mu * p[12] * nr))
    cf = math.sqrt(max(0.0, 1.0 - p[11] * uf * uf))
    cr = math.sqrt(max(0.0, 1.0 - p[11] * ur * ur))
    ext_f = p[4] * gf * p[42]
    fyf = math.copysign(1.0, af) * mu * sy * nf * _curve(abs(af), p[3], ext_f, p[5], min(p[6], ext_f)) * cf
    fyr = math.copysign(1.0, ar) * mu * sy * p[41] * nr * _curve(abs(ar), p[3], p[4], p[5], p[6]) * cr
    cw, sw = math.cos(wheel), math.sin(wheel)
    fx_b = fxf * cw - fyf * sw + fxr
    fy_b = fxf * sw + fyf * cw + fyr
    mz = lf * (fxf * sw + fyf * cw) - lr * fyr
    dvx = fx_b / m + vy * r - p[18] * vx
    dvy = fy_b / m - vx * r - p[18] * vy
    dr = mz / iz - p[19] * r
    return dvx, dvy, dr, af, ar


@njit(cache=True)
def simulate(st, p, ev_st, ev_sv, ev_tt, ev_tv, act, t_end, edt, ox, oy, res, check_walls):
    """Run physics until simulator time t_end. ev_* are command events (time sent, value), oldest first;
    act = [active steering cmd, active throttle cmd, next steer event, next throttle event] is updated
    in place. Returns the output vector (see O_*)."""
    out = np.zeros(6)
    half_l, half_w = 0.5 * CAR_LENGTH, 0.5 * CAR_WIDTH
    dt = DT_PHYS
    model = int(p[0])
    limit_deg = math.degrees(p[28])
    rate_deg = math.degrees(p[25])
    k = 0
    while st[S_T] < t_end - 1e-9:
        t = st[S_T]
        i = int(act[2])
        while i < ev_st.shape[0] and t >= ev_st[i] + p[26]:
            act[0] = ev_sv[i]
            i += 1
        act[2] = i
        i = int(act[3])
        while i < ev_tt.shape[0] and t >= ev_tt[i] + p[27]:
            act[1] = ev_tv[i]
            i += 1
        act[3] = i
        c_steer, thr = act[0], act[1]
        if model == 0:
            st[S_STEER] = _autodrive_steer(st[S_STEER], c_steer, rate_deg, limit_deg, dt)
            wheel = -math.radians(st[S_STEER])
            dvx, dvy, dr, af, ar = _twin_derivs(st, p, wheel, thr)
            vx = max(0.0, st[S_VX] + dt * dvx)
            vy = st[S_VY] + dt * dvy
            r = st[S_R] + dt * dr
            # below 2 m/s blend to the kinematic bicycle (stiff tyres, tiny speeds)
            w = min(1.0, max(0.0, vx - 1.0))
            if w < 1.0:
                L = p[22] + p[23]
                r_k = vx * math.tan(math.atan(p[10] * math.tan(wheel))) / L
                r = w * r + (1.0 - w) * r_k
                vy = w * vy + (1.0 - w) * p[23] * r_k
            st[S_AX] = dvx
            st[S_VX], st[S_VY], st[S_R] = vx, vy, r
            sf = abs(af) / p[3]
            sr = abs(ar) / p[3]
            if vx > 1.0:
                if sf > out[O_SF]:
                    out[O_SF] = sf
                if sr > out[O_SR]:
                    out[O_SR] = sr
                # body slip beyond what rolling tyres give (atan(l_r/L tan wheel) is geometry, not a spin)
                beta_kin = math.atan(p[23] / (p[22] + p[23]) * p[10] * math.tan(wheel))
                beta = abs(math.atan2(vy, vx) - beta_kin)
                if beta > out[O_BETA]:
                    out[O_BETA] = beta
                if sf > SLIDE_SLIP:
                    out[O_TSF] += dt
                if sr > SLIDE_SLIP:
                    out[O_TSR] += dt
        else:
            # ../env/race_env.py, step() and _grip(), per physics step instead of per 10 ms
            target = c_steer * p[28]
            wheel = -math.radians(st[S_STEER])
            stp = p[25] * dt
            wheel += min(stp, max(-stp, target - wheel))
            st[S_STEER] = -math.degrees(wheel)
            v = st[S_VX]
            L = WHEELBASE
            steer_eff = wheel
            if v >= 0.5:
                aw = abs(wheel)
                dem = v * v * math.tan(aw) / L
                if st[S_SLIDE] > 0.5:
                    if dem < p[34]:
                        st[S_SLIDE] = 0.0
                else:
                    if aw <= p[31]:
                        lim = p[29]
                    elif aw >= p[32]:
                        lim = p[30]
                    else:
                        lim = p[29] + (p[30] - p[29]) * (aw - p[31]) / (p[32] - p[31])
                    st[S_OVER] = st[S_OVER] + dt if dem > lim else 0.0
                    if st[S_OVER] >= p[33]:
                        st[S_SLIDE] = 1.0
                        st[S_OVER] = 0.0
                if st[S_SLIDE] > 0.5 and dem > p[34]:
                    steer_eff = math.copysign(math.atan(p[34] * L / (v * v)), wheel)
                else:
                    steer_eff = math.copysign(math.atan(p[10] * math.tan(aw)), wheel)
            else:
                st[S_SLIDE] = 0.0
            if thr > 0.0:
                acc = max(-p[37], min(p[36], p[39] * (p[40] * thr - v)))
            else:
                acc = -p[38] if v > 0.05 else -v / 0.01
            if v > 0.5:
                acc -= p[35] * abs(wheel)
            v = max(0.0, v + dt * acc)
            r = v * math.tan(steer_eff) / L
            # state is the centre of mass; the rear axle moves along the heading at v, as in the gym's KS model
            st[S_VX], st[S_VY], st[S_R], st[S_AX] = v, p[23] * r, r, acc
            if st[S_SLIDE] > 0.5:
                out[O_TSF] += dt
                out[O_SF] = max(out[O_SF], 3.0)
        # pose
        c, s = math.cos(st[S_PSI]), math.sin(st[S_PSI])
        st[S_X] += dt * (st[S_VX] * c - st[S_VY] * s)
        st[S_Y] += dt * (st[S_VX] * s + st[S_VY] * c)
        st[S_PSI] += dt * st[S_R]
        st[S_T] = t + dt
        k += 1
        if check_walls and k % COLLISION_EVERY == 0:
            if _wall_hit(st[S_X], st[S_Y], st[S_PSI], edt, ox, oy, res, half_l, half_w):
                out[O_COL] = 1.0
                return out
    return out


class Car:
    """One car: state, parameters and the command queues with AutoDRIVE's delays."""

    def __init__(self, params: np.ndarray):
        self.p = params
        self.st = new_state(0.0, 0.0, 0.0)
        self.reset(0.0, 0.0, 0.0, 0.0)

    def reset(self, x, y, yaw, speed=0.0, wheel=0.0):
        """x, y is the rear axle (the odometry frame); the state is kept at the centre of mass."""
        lr = self.p[IDX['lr']]
        self.st = new_state(x + lr * math.cos(yaw), y + lr * math.sin(yaw), yaw, speed)
        self.st[S_STEER] = -math.degrees(wheel)
        self.ev_s = [(-1.0, wheel / self.p[IDX['max_steer']])]
        self.ev_t = [(-1.0, 0.0)]
        self.act = np.array([wheel / self.p[IDX['max_steer']], 0.0, 0.0, 0.0])

    @property
    def t(self):
        return float(self.st[S_T])

    def command(self, steer_cmd: float, thr_cmd: float):
        """Send commands now (they arrive after the delays)."""
        self.ev_s.append((self.t, float(np.clip(steer_cmd, -1.0, 1.0))))
        self.ev_t.append((self.t, float(np.clip(thr_cmd, 0.0, 1.0))))

    def run(self, duration: float, edt=None, ox=0.0, oy=0.0, res=1.0) -> np.ndarray:
        es = np.array(self.ev_s, dtype=np.float64)
        et = np.array(self.ev_t, dtype=np.float64)
        self.act[2], self.act[3] = 0.0, 0.0
        check = edt is not None
        grid = edt if check else np.ones((1, 1))
        out = simulate(self.st, self.p, es[:, 0].copy(), es[:, 1].copy(), et[:, 0].copy(), et[:, 1].copy(),
                       self.act, self.t + duration, grid, ox, oy, res, check)
        self.ev_s = self.ev_s[max(0, int(self.act[2]) - 1):]       # keep the active one and what is pending
        self.ev_t = self.ev_t[max(0, int(self.act[3]) - 1):]
        return out

    # convenience
    @property
    def x(self):
        """Rear axle x, the odometry frame (F1TENTH base_link convention, and what the KS env used)."""
        return float(self.st[S_X] - self.p[IDX['lr']] * math.cos(self.st[S_PSI]))

    @property
    def y(self):
        return float(self.st[S_Y] - self.p[IDX['lr']] * math.sin(self.st[S_PSI]))

    @property
    def yaw(self):
        return float(self.st[S_PSI])

    @property
    def speed(self):
        """Forward speed in the body frame, the same everywhere on the car's axis (/odom twist.linear.x)."""
        return float(self.st[S_VX])

    @property
    def yaw_rate(self):
        return float(self.st[S_R])

    @property
    def wheel(self):
        return wheel_angle(self.st)
