export interface ParamSpec {
  name: string;
  label: string;
  /** Short hover tip explaining what this parameter changes. */
  info: string;
  min: number;
  max: number;
  step: number;
  default: number;
}

/** Metric shown in the right-hand Vehicle Status panel. */
export type StatusMetric =
  | "speed"
  | "steering"
  | "throttle"
  | "position"
  | "heading"
  | "lap"
  | "lap_time"
  | "best_lap"
  | "collisions"
  // wall follow
  | "wall_error"
  | "p_term"
  | "i_term"
  | "d_term"
  | "car_dist"
  // follow the gap
  | "best_point_range"
  | "closest_range"
  | "best_point_angle"
  | "gap_width"
  // pure pursuit / mpc
  | "target_velocity"
  | "speed_error"
  | "raceline_s"
  | "solve_ms"
  // rrt / rrt*
  | "plan_ms"
  | "tree_size"
  | "path_len"
  | "reached_goal";

/** Extra center-panel LiDAR overlays for an algorithm. */
export type VizFeature = "wall_rays" | "gap_target" | "rrt_path" | "rrt_tree";

/** Strip chart fed from one algo_debug signal. */
export interface ChartSpec {
  debugKey: string;
  label: string;
  subtitle: string;
  unit: string;
  /** Multiplier applied to the raw debug value (e.g. rad → deg). */
  scale?: number;
  warnAbs: number;
  minRange: number;
  decimals?: number;
}

export interface AlgorithmSpec {
  id: string;
  label: string;
  params: ParamSpec[];
  /** Shared + algorithm-specific status rows (order matters). */
  statusMetrics: StatusMetric[];
  /** Center-panel overlays. */
  vizFeatures: VizFeature[];
  /** Bottom strip chart (next to Path). */
  chart?: ChartSpec;
  /** UI group: reactive controllers, local planners, or map-based racing. */
  group: "reactive" | "local" | "map";
}

const SHARED_STATUS: StatusMetric[] = [
  "speed",
  "steering",
  "throttle",
  "position",
  "heading",
  "lap",
  "lap_time",
  "best_lap",
  "collisions",
];

export const ALGORITHMS: AlgorithmSpec[] = [
  {
    id: "wall_follow",
    label: "Wall Follow",
    group: "reactive",
    params: [
      {
        name: "kp",
        label: "Kp (proportional)",
        info: "How strongly steering corrects wall-distance error. Higher = snappier turns; too high causes oscillation.",
        min: 0,
        max: 20,
        step: 0.1,
        default: 6.8,
      },
      {
        name: "kd",
        label: "Kd (derivative)",
        info: "Damps rapid changes in error to reduce weave. Higher = smoother, but can lag behind the wall.",
        min: 0,
        max: 10,
        step: 0.1,
        default: 2.0,
      },
      {
        name: "ki",
        label: "Ki (integral)",
        info: "Accumulates steady-state offset from the wall setpoint. Usually leave at 0 unless the car drifts.",
        min: 0,
        max: 1,
        step: 0.005,
        default: 0.0,
      },
      {
        name: "desired_dist",
        label: "Wall distance (m)",
        info: "Target clearance from the left wall in meters. Larger = drives farther from the wall.",
        min: 0.3,
        max: 1.5,
        step: 0.05,
        default: 0.7,
      },
      {
        name: "lookahead",
        label: "Lookahead (m)",
        info: "How far ahead (m) the controller projects wall distance for anticipatory steering.",
        min: 0.1,
        max: 1.5,
        step: 0.05,
        default: 0.45,
      },
      {
        name: "throttle",
        label: "Throttle",
        info: "Normalized drive command (−1…1). Higher = faster; keep low while tuning.",
        min: 0,
        max: 0.5,
        step: 0.01,
        default: 0.05,
      },
    ],
    statusMetrics: [
      ...SHARED_STATUS.slice(0, 3),
      "wall_error",
      "car_dist",
      "p_term",
      "i_term",
      "d_term",
      ...SHARED_STATUS.slice(3),
    ],
    vizFeatures: ["wall_rays"],
    chart: {
      debugKey: "error",
      label: "Wall error",
      subtitle: "Desired − projected distance",
      unit: "m",
      warnAbs: 0.4,
      minRange: 0.2,
    },
  },
  {
    id: "follow_gap",
    label: "Follow the Gap",
    group: "reactive",
    params: [
      {
        name: "throttle",
        label: "Throttle",
        info: "Normalized drive command (−1…1). Higher = faster; keep low while tuning.",
        min: 0,
        max: 0.5,
        step: 0.01,
        default: 0.04,
      },
      {
        name: "vehicle_half_width",
        label: "Bubble half-width (m)",
        info: "Half-width of the safety bubble around the nearest obstacle. Larger = more clearance, narrower gaps.",
        min: 0.2,
        max: 1.0,
        step: 0.05,
        default: 0.55,
      },
      {
        name: "free_space_threshold",
        label: "Free space cutoff (m)",
        info: "Minimum LiDAR range for a beam to count as free when finding gaps. Higher = only deeper openings qualify.",
        min: 0.3,
        max: 3,
        step: 0.05,
        default: 0.9,
      },
      {
        name: "best_point_threshold",
        label: "Best point cutoff (m)",
        info: "Minimum range when picking the deepest sub-gap inside the chosen gap. Higher = aims farther down-track.",
        min: 0.5,
        max: 5,
        step: 0.1,
        default: 2.0,
      },
      {
        name: "disparity_threshold",
        label: "Disparity jump (m)",
        info: "Depth jump that triggers edge extension along obstacle sides. Larger = less aggressive widening of obstacles.",
        min: 0.5,
        max: 5,
        step: 0.1,
        default: 2.5,
      },
      {
        name: "smoothing_window",
        label: "Smoothing window (beams)",
        info: "Number of neighboring LiDAR beams averaged to reduce noise before gap finding.",
        min: 1,
        max: 50,
        step: 1,
        default: 25,
      },
      {
        name: "fov_deg",
        label: "Search FOV (deg)",
        info: "Angular field of view used when searching for the closest obstacle. Narrower = ignores side walls more.",
        min: 40,
        max: 270,
        step: 5,
        default: 100,
      },
      {
        name: "heading_weight",
        label: "Straight-ahead bias",
        info: "How strongly gap selection prefers openings straight ahead (0 = none, 1 = strong forward bias).",
        min: 0,
        max: 1,
        step: 0.05,
        default: 0.7,
      },
    ],
    statusMetrics: [
      ...SHARED_STATUS.slice(0, 3),
      "best_point_angle",
      "best_point_range",
      "closest_range",
      "gap_width",
      ...SHARED_STATUS.slice(3),
    ],
    vizFeatures: ["gap_target"],
    chart: {
      debugKey: "best_point_angle",
      label: "Gap heading",
      subtitle: "Target direction vs straight ahead",
      unit: "°",
      scale: 180 / Math.PI,
      warnAbs: 25,
      minRange: 10,
      decimals: 0,
    },
  },
  {
    id: "pure_pursuit",
    label: "Pure Pursuit",
    group: "map",
    params: [
      {
        name: "lookahead",
        label: "Lookahead (m)",
        info: "How far ahead on the raceline the target point sits. Larger = smoother but cuts corners; smaller = tighter tracking, may oscillate.",
        min: 0.3,
        max: 3.0,
        step: 0.05,
        default: 1.0,
      },
      {
        name: "velocity_scale",
        label: "Velocity scale",
        info: "Multiplier on the raceline's velocity targets. Start below 1.0 and raise as confidence grows.",
        min: 0.1,
        max: 1.5,
        step: 0.05,
        default: 1.0,
      },
      {
        name: "throttle_gain",
        label: "Throttle FF gain",
        info: "Feedforward map from target speed (m/s) to normalized throttle on straights. PI closes the residual.",
        min: 0.0,
        max: 0.12,
        step: 0.001,
        default: 0.043,
      },
      {
        name: "corner_throttle_gain",
        label: "Corner FF gain",
        info: "Feedforward gain when steering exceeds the corner threshold. Lower = safer, slower corner entry.",
        min: 0.0,
        max: 0.12,
        step: 0.001,
        default: 0.025,
      },
      {
        name: "corner_steering_threshold",
        label: "Corner threshold",
        info: "Normalized steering magnitude above which the corner feedforward gain applies.",
        min: 0.1,
        max: 1.0,
        step: 0.05,
        default: 0.6,
      },
      {
        name: "kp_speed",
        label: "Speed Kp",
        info: "Proportional gain on (v* − v) for the velocity PI that outputs throttle.",
        min: 0.0,
        max: 1.0,
        step: 0.01,
        default: 0.2,
      },
      {
        name: "ki_speed",
        label: "Speed Ki",
        info: "Integral gain on speed error. Removes steady-state lag vs the raceline velocity profile.",
        min: 0.0,
        max: 0.5,
        step: 0.005,
        default: 0.05,
      },
      {
        name: "integral_limit",
        label: "Integral limit",
        info: "Anti-windup clamp on the speed-error integral (|∫e dt| ≤ limit).",
        min: 0.1,
        max: 5.0,
        step: 0.1,
        default: 2.0,
      },
      {
        name: "max_steering_rad",
        label: "Max steering (rad)",
        info: "Steering angle limit used for normalization. Match the vehicle's physical limit.",
        min: 0.2,
        max: 1.0,
        step: 0.01,
        default: 0.58,
      },
      {
        name: "path_direction",
        label: "Path direction",
        info: "Travel sense along the raceline (+1 or −1). If the lookahead target sits behind the car, flip this (AutoDRIVE note).",
        min: -1,
        max: 1,
        step: 2,
        default: -1,
      },
      {
        name: "steering_direction",
        label: "Steering direction",
        info: "Sign flip for steering command. AutoDRIVE default is −1 (with an extra publish negation matching the Sim-Racing node).",
        min: -1,
        max: 1,
        step: 2,
        default: -1,
      },
    ],
    statusMetrics: [
      ...SHARED_STATUS.slice(0, 3),
      "target_velocity",
      "speed_error",
      "raceline_s",
      ...SHARED_STATUS.slice(3),
    ],
    vizFeatures: [],
    chart: {
      debugKey: "speed_error",
      label: "Speed error",
      subtitle: "Target − actual velocity",
      unit: "m/s",
      warnAbs: 1.5,
      minRange: 0.5,
      decimals: 1,
    },
  },
  {
    id: "mpc",
    label: "MPC",
    group: "map",
    params: [
      {
        name: "velocity_scale",
        label: "Velocity scale",
        info: "Multiplier on the raceline velocity profile before MPC tracking.",
        min: 0.1,
        max: 1.5,
        step: 0.05,
        default: 1.0,
      },
      {
        name: "max_speed",
        label: "Max speed (m/s)",
        info: "Hard speed limit inside the MPC QP.",
        min: 1.0,
        max: 8.0,
        step: 0.1,
        default: 6.0,
      },
      {
        name: "q_pos",
        label: "Q position",
        info: "State cost on (x, y) tracking error. Higher = sticks closer to the raceline.",
        min: 1,
        max: 40,
        step: 0.5,
        default: 13.5,
      },
      {
        name: "q_yaw",
        label: "Q yaw",
        info: "State cost on heading error.",
        min: 1,
        max: 40,
        step: 0.5,
        default: 13.0,
      },
      {
        name: "q_vel",
        label: "Q velocity",
        info: "State cost on speed-profile tracking error.",
        min: 0.5,
        max: 20,
        step: 0.5,
        default: 5.5,
      },
      {
        name: "r_accel",
        label: "R accel",
        info: "Input cost on longitudinal acceleration. Higher = smoother throttle changes.",
        min: 0.001,
        max: 1.0,
        step: 0.001,
        default: 0.01,
      },
      {
        name: "r_steer",
        label: "R steer",
        info: "Input cost on steering. Higher = less aggressive steering commands.",
        min: 1,
        max: 500,
        step: 1,
        default: 100,
      },
      {
        name: "throttle_gain",
        label: "Throttle FF gain",
        info: "Feedforward map from target speed (m/s) to normalized throttle on straights.",
        min: 0.0,
        max: 0.12,
        step: 0.001,
        default: 0.043,
      },
      {
        name: "corner_throttle_gain",
        label: "Corner FF gain",
        info: "Feedforward gain when steering exceeds the corner threshold.",
        min: 0.0,
        max: 0.12,
        step: 0.001,
        default: 0.025,
      },
      {
        name: "corner_steering_threshold",
        label: "Corner threshold",
        info: "Normalized steering magnitude above which the corner feedforward gain applies.",
        min: 0.1,
        max: 1.0,
        step: 0.05,
        default: 0.6,
      },
      {
        name: "kp_speed",
        label: "Speed Kp",
        info: "Proportional gain on (v* − v) for the velocity PI that outputs throttle.",
        min: 0.0,
        max: 1.0,
        step: 0.01,
        default: 0.2,
      },
      {
        name: "ki_speed",
        label: "Speed Ki",
        info: "Integral gain on speed error.",
        min: 0.0,
        max: 0.5,
        step: 0.005,
        default: 0.05,
      },
      {
        name: "integral_limit",
        label: "Integral limit",
        info: "Anti-windup clamp on the speed-error integral.",
        min: 0.1,
        max: 5.0,
        step: 0.1,
        default: 2.0,
      },
      {
        name: "max_steering_rad",
        label: "Max steering (rad)",
        info: "Steering angle limit used for normalization and the QP bound.",
        min: 0.2,
        max: 1.0,
        step: 0.01,
        default: 0.58,
      },
      {
        name: "path_direction",
        label: "Path direction",
        info: "Travel sense along the raceline (+1 or −1). Flip if the car tracks the wrong way.",
        min: -1,
        max: 1,
        step: 2,
        default: -1,
      },
      {
        name: "steering_direction",
        label: "Steering direction",
        info: "Sign flip for steering command. AutoDRIVE default is −1.",
        min: -1,
        max: 1,
        step: 2,
        default: -1,
      },
    ],
    statusMetrics: [
      ...SHARED_STATUS.slice(0, 3),
      "target_velocity",
      "speed_error",
      "raceline_s",
      "solve_ms",
      ...SHARED_STATUS.slice(3),
    ],
    vizFeatures: [],
    chart: {
      debugKey: "speed_error",
      label: "Speed error",
      subtitle: "Target − actual velocity",
      unit: "m/s",
      warnAbs: 1.5,
      minRange: 0.5,
      decimals: 1,
    },
  },
  {
    id: "rrt",
    label: "RRT",
    group: "local",
    params: [
      {
        name: "throttle",
        label: "Throttle",
        info: "Normalized drive command (−1…1). Higher = faster; keep low while tuning.",
        min: 0,
        max: 0.3,
        step: 0.01,
        default: 0.06,
      },
      {
        name: "goal_x",
        label: "Goal ahead (m)",
        info: "How far ahead the planner aims in the vehicle frame.",
        min: 1.0,
        max: 6.0,
        step: 0.1,
        default: 3.0,
      },
      {
        name: "lookahead_dist",
        label: "Lookahead (m)",
        info: "Pure Pursuit lookahead along the planned path.",
        min: 0.4,
        max: 2.5,
        step: 0.05,
        default: 1.4,
      },
      {
        name: "max_rrt_iters",
        label: "Max iterations",
        info: "Maximum tree expansions per plan cycle. Higher = better paths, more CPU.",
        min: 50,
        max: 500,
        step: 10,
        default: 220,
      },
      {
        name: "max_planning_ms",
        label: "Plan budget (ms)",
        info: "Maximum time allowed for one planning cycle.",
        min: 10,
        max: 100,
        step: 5,
        default: 40,
      },
      {
        name: "plan_hz",
        label: "Plan rate (Hz)",
        info: "How often the planner runs.",
        min: 1,
        max: 10,
        step: 0.5,
        default: 5,
      },
      {
        name: "expand_dist",
        label: "Step size (m)",
        info: "How far each tree extension grows toward a sample.",
        min: 0.05,
        max: 0.6,
        step: 0.05,
        default: 0.2,
      },
      {
        name: "goal_sample_rate",
        label: "Goal bias",
        info: "Probability of sampling the goal instead of free space (0–1).",
        min: 0,
        max: 0.5,
        step: 0.05,
        default: 0.2,
      },
      {
        name: "inflate_radius",
        label: "Inflate (m)",
        info: "Obstacle inflation on the LiDAR grid. Larger = more clearance.",
        min: 0.05,
        max: 0.35,
        step: 0.01,
        default: 0.1,
      },
    ],
    statusMetrics: [
      ...SHARED_STATUS.slice(0, 3),
      "plan_ms",
      "tree_size",
      "path_len",
      "reached_goal",
      ...SHARED_STATUS.slice(3),
    ],
    vizFeatures: ["rrt_path", "rrt_tree"],
    chart: {
      debugKey: "plan_ms",
      label: "Plan time",
      subtitle: "Planning cycle duration",
      unit: "ms",
      warnAbs: 40,
      minRange: 10,
      decimals: 1,
    },
  },
  {
    id: "rrt_star",
    label: "RRT*",
    group: "local",
    params: [
      {
        name: "throttle",
        label: "Throttle",
        info: "Normalized drive command (−1…1). Higher = faster; keep low while tuning.",
        min: 0,
        max: 0.3,
        step: 0.01,
        default: 0.06,
      },
      {
        name: "goal_x",
        label: "Goal ahead (m)",
        info: "How far ahead the planner aims in the vehicle frame.",
        min: 1.0,
        max: 6.0,
        step: 0.1,
        default: 3.0,
      },
      {
        name: "lookahead_dist",
        label: "Lookahead (m)",
        info: "Pure Pursuit lookahead along the planned path.",
        min: 0.4,
        max: 2.5,
        step: 0.05,
        default: 1.4,
      },
      {
        name: "max_rrt_iters",
        label: "Max iterations",
        info: "Maximum tree expansions per plan cycle. Higher = better paths, more CPU.",
        min: 50,
        max: 400,
        step: 10,
        default: 220,
      },
      {
        name: "max_planning_ms",
        label: "Plan budget (ms)",
        info: "Maximum time allowed for one planning cycle.",
        min: 10,
        max: 100,
        step: 5,
        default: 40,
      },
      {
        name: "plan_hz",
        label: "Plan rate (Hz)",
        info: "How often the planner runs.",
        min: 1,
        max: 10,
        step: 0.5,
        default: 5,
      },
      {
        name: "expand_dist",
        label: "Step size (m)",
        info: "How far each tree extension grows toward a sample.",
        min: 0.05,
        max: 0.6,
        step: 0.05,
        default: 0.2,
      },
      {
        name: "goal_sample_rate",
        label: "Goal bias",
        info: "Probability of sampling the goal instead of free space (0–1).",
        min: 0,
        max: 0.5,
        step: 0.05,
        default: 0.2,
      },
      {
        name: "inflate_radius",
        label: "Inflate (m)",
        info: "Obstacle inflation on the LiDAR grid. Larger = more clearance.",
        min: 0.05,
        max: 0.35,
        step: 0.01,
        default: 0.1,
      },
      {
        name: "near_radius",
        label: "Near radius (m)",
        info: "Neighborhood radius for parent selection and rewiring.",
        min: 0.3,
        max: 1.5,
        step: 0.05,
        default: 0.75,
      },
    ],
    statusMetrics: [
      ...SHARED_STATUS.slice(0, 3),
      "plan_ms",
      "tree_size",
      "path_len",
      "reached_goal",
      ...SHARED_STATUS.slice(3),
    ],
    vizFeatures: ["rrt_path", "rrt_tree"],
    chart: {
      debugKey: "plan_ms",
      label: "Plan time",
      subtitle: "Planning cycle duration",
      unit: "ms",
      warnAbs: 40,
      minRange: 10,
      decimals: 1,
    },
  },
];

export function algorithmSpec(id: string | null | undefined): AlgorithmSpec {
  if (!id) return ALGORITHMS[0];
  return ALGORITHMS.find((a) => a.id === id) ?? ALGORITHMS[0];
}

export function algorithmsInGroup(
  group: AlgorithmSpec["group"]
): AlgorithmSpec[] {
  return ALGORITHMS.filter((a) => a.group === group);
}

/** Raceline + AMCL controllers used by Mapping step 3. */
export function isMapControllerAlgo(id: string | null | undefined): boolean {
  return id === "pure_pursuit" || id === "mpc";
}
