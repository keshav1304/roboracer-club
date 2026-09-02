"use client";

import type { ReactNode } from "react";
import type { Telemetry } from "@/lib/useGateway";
import type { StatusMetric } from "@/lib/algorithms";

interface Props {
  telemetry: Telemetry;
  metrics: StatusMetric[];
}

function num(debug: Telemetry["algo_debug"], key: string): number | null {
  if (!debug) return null;
  const v = debug[key];
  return typeof v === "number" ? v : null;
}

function signedBar(value: number, color = "var(--accent)") {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <div className="bar-wrap">
        <div
          className="bar"
          style={{
            left: value < 0 ? `${50 + value * 50}%` : "50%",
            width: `${Math.min(Math.abs(value), 1) * 50}%`,
            background: color,
          }}
        />
      </div>
      <span className="v">{value.toFixed(2)}</span>
    </div>
  );
}

function pidBar(value: number | null) {
  if (value == null) return <span className="v">—</span>;
  // Scale loosely; PID terms can be large before steering normalization.
  const norm = Math.max(-1, Math.min(1, value / 5));
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <div className="bar-wrap">
        <div
          className="bar"
          style={{
            left: norm < 0 ? `${50 + norm * 50}%` : "50%",
            width: `${Math.abs(norm) * 50}%`,
            background:
              Math.abs(value) > 2 ? "var(--warn)" : "var(--accent)",
          }}
        />
      </div>
      <span className="v">{value.toFixed(2)}</span>
    </div>
  );
}

export default function RobotStatus({ telemetry, metrics }: Props) {
  const headingDeg = ((telemetry.yaw * 180) / Math.PI).toFixed(1);
  const [x, y] = telemetry.position;
  const debug = telemetry.algo_debug;

  const rows: Record<StatusMetric, { label: string; node: ReactNode }> = {
    speed: {
      label: "Speed",
      node: <span className="v">{telemetry.speed.toFixed(2)} m/s</span>,
    },
    steering: {
      label: "Steering",
      node: signedBar(telemetry.steering),
    },
    throttle: {
      label: "Throttle",
      node: signedBar(telemetry.throttle),
    },
    wall_error: (() => {
      const err = num(debug, "error");
      return {
        label: "Wall error",
        node:
          err == null ? (
            <span className="v">—</span>
          ) : (
            <span className={`v ${Math.abs(err) > 0.4 ? "warn" : "good"}`}>
              {err.toFixed(3)} m
            </span>
          ),
      };
    })(),
    car_dist: (() => {
      const d = num(debug, "car_dist");
      return {
        label: "Dist to wall",
        node:
          d == null ? (
            <span className="v">—</span>
          ) : (
            <span className="v">{d.toFixed(3)} m</span>
          ),
      };
    })(),
    p_term: {
      label: "P term",
      node: pidBar(num(debug, "p_term")),
    },
    i_term: {
      label: "I term",
      node: pidBar(num(debug, "i_term")),
    },
    d_term: {
      label: "D term",
      node: pidBar(num(debug, "d_term")),
    },
    best_point_angle: (() => {
      const a = num(debug, "best_point_angle");
      return {
        label: "Gap heading",
        node:
          a == null ? (
            <span className="v">—</span>
          ) : (
            <span className="v">{((a * 180) / Math.PI).toFixed(1)}°</span>
          ),
      };
    })(),
    best_point_range: (() => {
      const r = num(debug, "best_point_range");
      return {
        label: "Gap depth",
        node:
          r == null ? (
            <span className="v">—</span>
          ) : (
            <span className="v">{r.toFixed(2)} m</span>
          ),
      };
    })(),
    gap_width: (() => {
      const start = num(debug, "gap_start_angle");
      const end = num(debug, "gap_end_angle");
      const widthDeg =
        start != null && end != null ? ((end - start) * 180) / Math.PI : null;
      return {
        label: "Gap width",
        node:
          widthDeg == null ? (
            <span className="v">—</span>
          ) : (
            <span className={`v ${Math.abs(widthDeg) < 15 ? "warn" : ""}`}>
              {Math.abs(widthDeg).toFixed(1)}°
            </span>
          ),
      };
    })(),
    closest_range: (() => {
      const r = num(debug, "closest_range");
      return {
        label: "Closest obst.",
        node:
          r == null ? (
            <span className="v">—</span>
          ) : (
            <span className={`v ${r < 0.5 ? "bad" : r < 1.0 ? "warn" : "good"}`}>
              {r.toFixed(2)} m
            </span>
          ),
      };
    })(),
    target_velocity: (() => {
      const v = num(debug, "target_velocity");
      return {
        label: "Target speed",
        node:
          v == null ? (
            <span className="v">—</span>
          ) : (
            <span className="v">{v.toFixed(2)} m/s</span>
          ),
      };
    })(),
    speed_error: (() => {
      const e = num(debug, "speed_error");
      return {
        label: "Speed error",
        node:
          e == null ? (
            <span className="v">—</span>
          ) : (
            <span className={`v ${Math.abs(e) > 1.5 ? "warn" : "good"}`}>
              {e.toFixed(2)} m/s
            </span>
          ),
      };
    })(),
    raceline_s: (() => {
      const s = num(debug, "s_curr");
      return {
        label: "Track position",
        node:
          s == null ? (
            <span className="v">—</span>
          ) : (
            <span className="v">{s.toFixed(1)} m</span>
          ),
      };
    })(),
    solve_ms: (() => {
      const ms = num(debug, "solve_ms");
      return {
        label: "MPC solve",
        node:
          ms == null ? (
            <span className="v">—</span>
          ) : (
            <span className={`v ${ms > 40 ? "warn" : "good"}`}>
              {ms.toFixed(1)} ms
            </span>
          ),
      };
    })(),
    plan_ms: (() => {
      const ms = num(debug, "plan_ms");
      return {
        label: "Plan time",
        node:
          ms == null ? (
            <span className="v">—</span>
          ) : (
            <span className={`v ${ms > 40 ? "warn" : "good"}`}>
              {ms.toFixed(1)} ms
            </span>
          ),
      };
    })(),
    tree_size: (() => {
      const n = num(debug, "tree_size");
      return {
        label: "Tree nodes",
        node:
          n == null ? <span className="v">—</span> : <span className="v">{n.toFixed(0)}</span>,
      };
    })(),
    path_len: (() => {
      const n = num(debug, "path_len");
      return {
        label: "Path nodes",
        node:
          n == null ? <span className="v">—</span> : <span className="v">{n.toFixed(0)}</span>,
      };
    })(),
    reached_goal: (() => {
      const g = num(debug, "reached_goal");
      return {
        label: "Goal reached",
        node:
          g == null ? (
            <span className="v">—</span>
          ) : (
            <span className={`v ${g >= 0.5 ? "good" : "warn"}`}>
              {g >= 0.5 ? "yes" : "no"}
            </span>
          ),
      };
    })(),
    position: {
      label: "Position",
      node: (
        <span className="v">
          {x.toFixed(2)}, {y.toFixed(2)}
        </span>
      ),
    },
    heading: {
      label: "Heading",
      node: <span className="v">{headingDeg}°</span>,
    },
    lap: {
      label: "Lap",
      node: <span className="v">{telemetry.lap_count}</span>,
    },
    lap_time: {
      label: "Lap time",
      node: <span className="v">{telemetry.lap_time.toFixed(2)} s</span>,
    },
    best_lap: {
      label: "Best lap",
      node: (
        <span className="v good">{telemetry.best_lap_time.toFixed(2)} s</span>
      ),
    },
    collisions: {
      label: "Collisions",
      node: (
        <span className={`v ${telemetry.collision_count > 0 ? "bad" : ""}`}>
          {telemetry.collision_count}
        </span>
      ),
    },
  };

  return (
    <div style={{ flex: 1 }}>
      {metrics.map((key) => {
        const row = rows[key];
        if (!row) return null;
        return (
          <div className="metric-row" key={key}>
            <span className="k">{row.label}</span>
            {row.node}
          </div>
        );
      })}
    </div>
  );
}
