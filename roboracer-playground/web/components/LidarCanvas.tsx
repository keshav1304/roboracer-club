"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";
import type { AlgoDebug, LidarFrame, Telemetry } from "@/lib/useGateway";
import type { VizFeature } from "@/lib/algorithms";

/** Display range for the LiDAR panel (meters). Smaller = more zoomed in. */
const LIDAR_DISPLAY_RANGE_M = 5;

interface Props {
  lidar: LidarFrame | null;
  telemetry: Telemetry;
  trail: [number, number][];
  vizFeatures: VizFeature[];
  /** Optional bottom-right panel (e.g. wall-error chart). */
  bottomExtra?: ReactNode;
  /** Clear the world-frame path trail. */
  onClearTrail?: () => void;
}

const COLORS = {
  grid: "#dfe5ef",
  gridLabel: "#98a2b3",
  points: "#3b76e8",
  car: "#0ea36f",
  steer: "#d97706",
  trail: "#3b76e8",
  rayA: "#7c3aed",
  rayB: "#db2777",
  wall: "#0ea36f",
  desired: "rgba(14, 163, 111, 0.18)",
  lookahead: "#d97706",
  gapArc: "rgba(59, 118, 232, 0.2)",
  gapTarget: "#dc2626",
  rrtTree: "rgba(59, 118, 232, 0.45)",
  rrtPath: "#ea580c",
  rrtGoal: "#7c3aed",
  occHit: "rgba(220, 38, 38, 0.75)",
  occInflated: "rgba(254, 240, 138, 0.7)",
};

function setupCanvas(canvas: HTMLCanvasElement) {
  const dpr = window.devicePixelRatio || 1;
  const box = canvas.parentElement;
  const w = Math.floor(box?.clientWidth || canvas.clientWidth);
  const h = Math.floor(box?.clientHeight || canvas.clientHeight);
  if (w < 2 || h < 2) return null;
  const bw = Math.max(1, Math.floor(w * dpr));
  const bh = Math.max(1, Math.floor(h * dpr));
  if (canvas.width !== bw || canvas.height !== bh) {
    canvas.width = bw;
    canvas.height = bh;
  }
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

/** Map vehicle-frame polar (angle, range) → canvas, matching LiDAR points. */
function polarToCanvas(
  cx: number,
  cy: number,
  scale: number,
  angle: number,
  range: number
) {
  return {
    x: cx - Math.sin(angle) * range * scale,
    y: cy - Math.cos(angle) * range * scale,
  };
}

function drawWallRays(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  scale: number,
  debug: AlgoDebug
) {
  const rayA = Number(debug.ray_a);
  const rayB = Number(debug.ray_b);
  const theta = Number(debug.theta_rad);
  const alpha = Number(debug.alpha);
  const desired = Number(debug.desired_dist);
  const lookahead = Number(debug.lookahead);
  const carDist = Number(debug.car_dist);
  if (![rayA, rayB, theta, alpha, desired].every(Number.isFinite)) return;

  const rayBAngle = Math.PI / 2;
  const rayAAngle = rayBAngle - theta;

  // Desired-distance arc on the left
  ctx.beginPath();
  ctx.strokeStyle = COLORS.wall;
  ctx.lineWidth = 1.5;
  ctx.setLineDash([4, 3]);
  ctx.arc(cx, cy, desired * scale, -Math.PI * 0.15, -Math.PI * 0.85, true);
  ctx.stroke();
  ctx.setLineDash([]);

  // Soft fill band toward the wall
  ctx.fillStyle = COLORS.desired;
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.arc(cx, cy, desired * scale, -Math.PI * 0.15, -Math.PI * 0.85, true);
  ctx.closePath();
  ctx.fill();

  // Ray B (90° left)
  const bEnd = polarToCanvas(cx, cy, scale, rayBAngle, rayB);
  ctx.strokeStyle = COLORS.rayB;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(bEnd.x, bEnd.y);
  ctx.stroke();
  ctx.fillStyle = COLORS.rayB;
  ctx.beginPath();
  ctx.arc(bEnd.x, bEnd.y, 3.5, 0, Math.PI * 2);
  ctx.fill();

  // Ray A (angled forward)
  const aEnd = polarToCanvas(cx, cy, scale, rayAAngle, rayA);
  ctx.strokeStyle = COLORS.rayA;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(aEnd.x, aEnd.y);
  ctx.stroke();
  ctx.fillStyle = COLORS.rayA;
  ctx.beginPath();
  ctx.arc(aEnd.x, aEnd.y, 3.5, 0, Math.PI * 2);
  ctx.fill();

  // Inferred wall segment through the two hit points
  if (Number.isFinite(carDist) && carDist > 0.05) {
    ctx.strokeStyle = COLORS.wall;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(aEnd.x, aEnd.y);
    ctx.lineTo(bEnd.x, bEnd.y);
    const dx = bEnd.x - aEnd.x;
    const dy = bEnd.y - aEnd.y;
    const len = Math.hypot(dx, dy) || 1;
    ctx.lineTo(bEnd.x + (dx / len) * 24, bEnd.y + (dy / len) * 24);
    ctx.stroke();
  }

  // Lookahead projection point (along heading, offset by projected wall distance)
  if (Number.isFinite(lookahead) && lookahead > 0) {
    const future = Number(debug.car_dist_future);
    const lx = cx - Math.sin(0) * lookahead * scale; // forward
    const ly = cy - Math.cos(0) * lookahead * scale;
    // Offset left by projected distance
    if (Number.isFinite(future)) {
      const px = lx - Math.sin(Math.PI / 2) * future * scale;
      const py = ly - Math.cos(Math.PI / 2) * future * scale;
      ctx.strokeStyle = COLORS.lookahead;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(lx, ly);
      ctx.lineTo(px, py);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = COLORS.lookahead;
      ctx.beginPath();
      ctx.arc(px, py, 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

function drawGapTarget(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  scale: number,
  debug: AlgoDebug
) {
  const start = Number(debug.gap_start_angle);
  const end = Number(debug.gap_end_angle);
  const target = Number(debug.best_point_angle);
  const range = Number(debug.best_point_range);
  const fov = Number(debug.fov_deg);

  if (Number.isFinite(start) && Number.isFinite(end)) {
    // Gap wedge (convert ROS angles to canvas arcs: our polar uses same angle)
    const r = Math.max(range || 3, 2) * scale;
    ctx.fillStyle = COLORS.gapArc;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    // Sweep from start to end in vehicle polar canvas mapping
    const steps = 24;
    for (let i = 0; i <= steps; i++) {
      const a = start + ((end - start) * i) / steps;
      const p = polarToCanvas(cx, cy, 1, a, r);
      ctx.lineTo(p.x, p.y);
    }
    ctx.closePath();
    ctx.fill();
  }

  if (Number.isFinite(fov) && fov > 0) {
    const half = ((fov / 2) * Math.PI) / 180;
    ctx.strokeStyle = "rgba(152, 162, 179, 0.6)";
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 2]);
    for (const a of [-half, half]) {
      const p = polarToCanvas(cx, cy, scale, a, 8);
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(p.x, p.y);
      ctx.stroke();
    }
    ctx.setLineDash([]);
  }

  if (Number.isFinite(target)) {
    const len = Number.isFinite(range) && range > 0.1 ? range : 4;
    const p = polarToCanvas(cx, cy, scale, target, len);
    ctx.strokeStyle = COLORS.gapTarget;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(p.x, p.y);
    ctx.stroke();
    ctx.fillStyle = COLORS.gapTarget;
    ctx.beginPath();
    ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
    ctx.fill();
  }
}

/** Vehicle-frame cartesian (x forward, y left) → canvas, matching LiDAR. */
function localXyToCanvas(
  cx: number,
  cy: number,
  scale: number,
  x: number,
  y: number
) {
  return {
    x: cx - y * scale,
    y: cy - x * scale,
  };
}

/** Map path cost t∈[0,1] → red → orange → yellow → green. */
function costToColor(t: number): string {
  const u = Math.max(0, Math.min(1, t));
  const stops: [number, number, number][] = [
    [220, 38, 38], // red (low cost)
    [249, 115, 22], // orange
    [234, 179, 8], // yellow
    [22, 163, 74], // green (high cost)
  ];
  const x = u * (stops.length - 1);
  const i = Math.min(Math.floor(x), stops.length - 2);
  const f = x - i;
  const a = stops[i];
  const b = stops[i + 1];
  const r = Math.round(a[0] + (b[0] - a[0]) * f);
  const g = Math.round(a[1] + (b[1] - a[1]) * f);
  const bl = Math.round(a[2] + (b[2] - a[2]) * f);
  return `rgb(${r}, ${g}, ${bl})`;
}

function drawOccCells(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  scale: number,
  cells: number[][] | undefined,
  fill: string,
  cellM: number,
  stroke?: string
) {
  if (!Array.isArray(cells) || cells.length === 0) return;
  // Keep cells readable even when the grid resolution is small.
  const size = Math.max(cellM * scale, 4);
  const half = size / 2;
  ctx.fillStyle = fill;
  for (const pt of cells) {
    if (!Array.isArray(pt) || pt.length < 2) continue;
    const [x, y] = pt;
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    const p = localXyToCanvas(cx, cy, scale, x, y);
    ctx.fillRect(p.x - half, p.y - half, size, size);
  }
  if (stroke) {
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1;
    for (const pt of cells) {
      if (!Array.isArray(pt) || pt.length < 2) continue;
      const [x, y] = pt;
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      const p = localXyToCanvas(cx, cy, scale, x, y);
      ctx.strokeRect(p.x - half, p.y - half, size, size);
    }
  }
}

function drawRrtOccupancy(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  scale: number,
  debug: AlgoDebug
) {
  const res = Number(debug.grid_resolution);
  const cellM = Number.isFinite(res) && res > 0 ? res : 0.05;
  // Inflate ring first, then raw hits on top.
  drawOccCells(
    ctx,
    cx,
    cy,
    scale,
    debug.occ_inflated,
    COLORS.occInflated,
    cellM,
    "rgba(202, 138, 4, 0.55)"
  );
  drawOccCells(ctx, cx, cy, scale, debug.occ_hits, COLORS.occHit, cellM);
}

function drawRrtTree(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  scale: number,
  debug: AlgoDebug
) {
  const tree = debug.tree;
  if (!Array.isArray(tree) || tree.length === 0) return;

  let maxCost = 0;
  let minCost = Infinity;
  for (const seg of tree) {
    if (!Array.isArray(seg) || seg.length < 5) continue;
    const c = Number(seg[4]);
    if (!Number.isFinite(c)) continue;
    if (c > maxCost) maxCost = c;
    if (c < minCost) minCost = c;
  }
  if (!Number.isFinite(minCost)) minCost = 0;
  if (maxCost <= minCost) maxCost = minCost + 1;

  ctx.lineWidth = 1.6;
  ctx.lineCap = "round";
  for (const seg of tree) {
    if (!Array.isArray(seg) || seg.length < 4) continue;
    const [x1, y1, x2, y2] = seg;
    if (![x1, y1, x2, y2].every(Number.isFinite)) continue;
    const cost = seg.length >= 5 ? Number(seg[4]) : minCost;
    const t = Number.isFinite(cost)
      ? (cost - minCost) / (maxCost - minCost)
      : 0;
    ctx.strokeStyle = costToColor(t);
    const a = localXyToCanvas(cx, cy, scale, x1, y1);
    const b = localXyToCanvas(cx, cy, scale, x2, y2);
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }
}

function drawRrtPath(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  scale: number,
  debug: AlgoDebug
) {
  const path = debug.path;
  if (Array.isArray(path) && path.length >= 2) {
    ctx.strokeStyle = COLORS.rrtPath;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    for (let i = 0; i < path.length; i++) {
      const pt = path[i];
      if (!Array.isArray(pt) || pt.length < 2) continue;
      const [x, y] = pt;
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      const p = localXyToCanvas(cx, cy, scale, x, y);
      if (i === 0) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
    }
    ctx.stroke();
  }

  const tx = Number(debug.target_x);
  const ty = Number(debug.target_y);
  if (Number.isFinite(tx) && Number.isFinite(ty)) {
    const p = localXyToCanvas(cx, cy, scale, tx, ty);
    ctx.fillStyle = COLORS.rrtPath;
    ctx.beginPath();
    ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
    ctx.fill();
  }

  const gx = Number(debug.goal_x);
  if (Number.isFinite(gx) && gx > 0) {
    const p = localXyToCanvas(cx, cy, scale, gx, 0);
    const tol = Number(debug.goal_tolerance);
    if (Number.isFinite(tol) && tol > 0) {
      ctx.strokeStyle = COLORS.rrtGoal;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.arc(p.x, p.y, tol * scale, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "rgba(124, 58, 237, 0.12)";
      ctx.beginPath();
      ctx.arc(p.x, p.y, tol * scale, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.strokeStyle = COLORS.rrtGoal;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(p.x, p.y);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = COLORS.rrtGoal;
    ctx.beginPath();
    ctx.arc(p.x, p.y, 3.5, 0, Math.PI * 2);
    ctx.fill();
  }
}

/** LiDAR scan in the vehicle frame + pose trail in the world frame,
 *  with algorithm-specific overlays. */
export default function LidarCanvas({
  lidar,
  telemetry,
  trail,
  vizFeatures,
  bottomExtra,
  onClearTrail,
}: Props) {
  const scanRef = useRef<HTMLCanvasElement>(null);
  const trailRef = useRef<HTMLCanvasElement>(null);
  const [showTree, setShowTree] = useState(true);
  const stateRef = useRef({ lidar, telemetry, trail, vizFeatures, showTree });
  stateRef.current = { lidar, telemetry, trail, vizFeatures, showTree };

  // --- LiDAR scan (vehicle frame) ---
  useEffect(() => {
    const canvas = scanRef.current;
    if (!canvas) return;
    let raf = 0;
    const render = () => {
      raf = requestAnimationFrame(render);
      const setup = setupCanvas(canvas);
      if (!setup) return;
      const { ctx, w, h } = setup;
      const { lidar, telemetry, vizFeatures, showTree } = stateRef.current;

      const cx = w / 2;
      const cy = h * 0.58;
      // Fill most of the panel; zoom by mapping a short display range.
      const maxR = Math.min(w, h) * 0.48;
      const displayRange = LIDAR_DISPLAY_RANGE_M;
      const scale = maxR / displayRange;

      ctx.strokeStyle = COLORS.grid;
      ctx.fillStyle = COLORS.gridLabel;
      ctx.font = "10px ui-monospace, monospace";
      for (const rm of [1, 2, 3, 4, 5]) {
        ctx.beginPath();
        ctx.arc(cx, cy, rm * scale, 0, Math.PI * 2);
        ctx.stroke();
        ctx.fillText(`${rm}m`, cx + rm * scale + 3, cy - 2);
      }

      if (lidar) {
        ctx.fillStyle = COLORS.points;
        const { angle_min, angle_increment, ranges } = lidar;
        for (let i = 0; i < ranges.length; i++) {
          const r = ranges[i];
          if (r <= 0.06 || r > displayRange) continue;
          const a = angle_min + i * angle_increment;
          const px = cx - Math.sin(a) * r * scale;
          const py = cy - Math.cos(a) * r * scale;
          ctx.fillRect(px - 1.5, py - 1.5, 3, 3);
        }
      }

      const debug = telemetry.algo_debug;
      if (debug) {
        if (vizFeatures.includes("wall_rays")) {
          drawWallRays(ctx, cx, cy, scale, debug);
        }
        if (vizFeatures.includes("gap_target")) {
          drawGapTarget(ctx, cx, cy, scale, debug);
        }
        if (
          vizFeatures.includes("rrt_path") ||
          vizFeatures.includes("rrt_tree")
        ) {
          drawRrtOccupancy(ctx, cx, cy, scale, debug);
        }
        if (vizFeatures.includes("rrt_tree") && showTree) {
          drawRrtTree(ctx, cx, cy, scale, debug);
        }
        if (vizFeatures.includes("rrt_path")) {
          drawRrtPath(ctx, cx, cy, scale, debug);
        }
      }

      // car marker + steering direction
      ctx.save();
      ctx.translate(cx, cy);
      ctx.fillStyle = COLORS.car;
      ctx.beginPath();
      ctx.moveTo(0, -10);
      ctx.lineTo(6, 8);
      ctx.lineTo(-6, 8);
      ctx.closePath();
      ctx.fill();
      const steerAngle = telemetry.steering * 0.5236;
      ctx.strokeStyle = COLORS.steer;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(-Math.sin(steerAngle) * 28, -Math.cos(steerAngle) * 28);
      ctx.stroke();
      ctx.restore();
    };
    raf = requestAnimationFrame(render);
    return () => cancelAnimationFrame(raf);
  }, []);

  // --- Pose trail (world frame) ---
  useEffect(() => {
    const canvas = trailRef.current;
    if (!canvas) return;
    let raf = 0;
    let alive = true;
    const render = () => {
      if (!alive) return;
      raf = requestAnimationFrame(render);
      const setup = setupCanvas(canvas);
      if (!setup) return;
      const { ctx, w, h } = setup;
      const { telemetry, trail } = stateRef.current;

      // Empty state is the HTML `.canvas-placeholder` overlay — keep the
      // canvas blank so the message is not painted twice.
      if (trail.length < 2) {
        return;
      }

      let minX = Infinity,
        maxX = -Infinity,
        minY = Infinity,
        maxY = -Infinity;
      for (const [x, y] of trail) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
      const spanX = Math.max(maxX - minX, 1);
      const spanY = Math.max(maxY - minY, 1);
      // Guard: when the canvas is shorter than the padding, avoid a negative scale
      // that flings the trail off-screen (regression from the layout-sizing fix).
      const s = Math.max(Math.min(w, h) - 48, 1) / Math.max(spanX, spanY);
      const ox = w / 2 - ((minX + maxX) / 2) * s;
      const oy = h / 2 + ((minY + maxY) / 2) * s;

      ctx.strokeStyle = COLORS.grid;
      ctx.lineWidth = 1;
      const step = 40;
      for (let gx = 0; gx <= w; gx += step) {
        ctx.beginPath();
        ctx.moveTo(gx, 0);
        ctx.lineTo(gx, h);
        ctx.stroke();
      }
      for (let gy = 0; gy <= h; gy += step) {
        ctx.beginPath();
        ctx.moveTo(0, gy);
        ctx.lineTo(w, gy);
        ctx.stroke();
      }

      ctx.strokeStyle = COLORS.trail;
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      trail.forEach(([x, y], i) => {
        const px = ox + x * s;
        const py = oy - y * s;
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.stroke();

      const [cxp, cyp] = telemetry.position;
      const px = ox + cxp * s;
      const py = oy - cyp * s;
      ctx.save();
      ctx.translate(px, py);
      ctx.rotate(-telemetry.yaw + Math.PI / 2);
      ctx.fillStyle = COLORS.car;
      ctx.beginPath();
      ctx.moveTo(0, -8);
      ctx.lineTo(5, 6);
      ctx.lineTo(-5, 6);
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    };
    raf = requestAnimationFrame(render);
    return () => {
      alive = false;
      cancelAnimationFrame(raf);
    };
  }, []);

  const hasRays = vizFeatures.includes("wall_rays");
  const hasGap = vizFeatures.includes("gap_target");
  const hasRrtTree = vizFeatures.includes("rrt_tree");
  const hasRrtPath = vizFeatures.includes("rrt_path");

  const legend: {
    color: string;
    label: string;
    dashed?: boolean;
    wide?: boolean;
  }[] = [
    { color: COLORS.points, label: "LiDAR hits" },
    { color: COLORS.car, label: "Car" },
    { color: COLORS.steer, label: "Steering" },
  ];
  if (hasRays) {
    legend.push(
      { color: COLORS.rayB, label: "Ray B (90° left)" },
      { color: COLORS.rayA, label: "Ray A (angled)" },
      { color: COLORS.wall, label: "Wall estimate" },
      { color: COLORS.wall, label: "Desired distance", dashed: true },
      { color: COLORS.lookahead, label: "Lookahead point", dashed: true }
    );
  }
  if (hasGap) {
    legend.push(
      { color: COLORS.gapTarget, label: "Target heading" },
      { color: "rgba(59, 118, 232, 0.45)", label: "Chosen gap" },
      { color: COLORS.gridLabel, label: "Search FOV", dashed: true }
    );
  }
  if (hasRrtTree || hasRrtPath) {
    legend.push(
      { color: COLORS.occHit, label: "Occupied cells" },
      { color: COLORS.occInflated, label: "Inflated cells" }
    );
  }
  if (hasRrtTree && showTree) {
    legend.push({
      color: "linear-gradient(90deg, #dc2626, #f97316, #eab308, #16a34a)",
      label: "Tree (low→high cost)",
      wide: true,
    });
  }
  if (hasRrtPath) {
    legend.push(
      { color: COLORS.rrtPath, label: "Planned path" },
      { color: COLORS.rrtGoal, label: "Goal + tolerance", dashed: true }
    );
  }

  return (
    <div className={`viz-board${bottomExtra ? " has-extra" : ""}`}>
      <div className="viz-card viz-lidar">
        <div className="viz-title">
          LiDAR <span>Vehicle frame</span>
          {hasRrtTree && (
            <label className="viz-toggle">
              <input
                type="checkbox"
                checked={showTree}
                onChange={(e) => setShowTree(e.target.checked)}
              />
              Tree
            </label>
          )}
          <span className="viz-meta">
            {lidar ? `${lidar.ranges.length} beams` : "No data"}
          </span>
        </div>
        <div className="canvas-wrap">
          <canvas ref={scanRef} className="viz" />
        </div>
        <div className="viz-legend">
          {legend.map((item) => (
            <span className="legend-item" key={item.label}>
              <span
                className={`legend-swatch${item.dashed ? " dashed" : ""}${item.wide ? " wide" : ""}`}
                style={
                  item.dashed
                    ? { borderColor: item.color }
                    : { background: item.color }
                }
              />
              {item.label}
            </span>
          ))}
        </div>
      </div>
      <div className="viz-bottom">
        <div className="viz-card viz-path">
          <div className="viz-title">
            Path <span>World frame</span>
            <span className="viz-meta">
              {trail.length > 1 ? `${trail.length} points` : "No data"}
            </span>
            {onClearTrail && (
              <button
                type="button"
                className="btn small"
                disabled={trail.length === 0}
                onClick={onClearTrail}
              >
                Clear
              </button>
            )}
          </div>
          <div className="canvas-wrap">
            <canvas ref={trailRef} className="viz" />
            {trail.length < 2 && (
              <div className="canvas-placeholder">Drive to draw the path</div>
            )}
          </div>
        </div>
        {bottomExtra}
      </div>
    </div>
  );
}
