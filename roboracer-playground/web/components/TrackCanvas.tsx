"use client";

import {
  useCallback,
  useEffect,
  useRef,
  type ReactNode,
} from "react";
import type { Telemetry, Transform2D } from "@/lib/useGateway";
import type { EditTool } from "@/lib/useMapping";
import { decodeOccB64, occupancyToCanvas } from "@/lib/occupancyMap";

/** Identity of a map image to render (live SLAM frame or saved map). */
export interface TrackSource {
  /** Unique per image content (name+version, or live frame seq). */
  key: string;
  /** Live occupancy (preferred). */
  occupancy?: Int8Array;
  /** REST URL returning { occ_b64, width, height, ... } for saved maps. */
  gridUrl?: string;
  /** Optional PNG thumbnail URL (fallback only). */
  url?: string;
  width: number;
  height: number;
  resolution: number;
  origin: [number, number];
  mapToWorld?: Transform2D | null;
  /** Identity of the underlying track — the view re-fits when it changes. */
  fitKey: string;
}

export interface RacelineOverlay {
  x: number[];
  y: number[];
  v: number[];
  clearance?: number[];
  margin?: number;
}

export type TrackMode = "view" | "edit-map" | "edit-line";

interface Props {
  source: TrackSource | null;
  telemetry: Telemetry;
  trail: [number, number][];
  raceline?: RacelineOverlay | null;
  /** Dashed preview polyline (optimizer progress). */
  preview?: { x: number[]; y: number[] } | null;
  anchors?: [number, number][];
  selectedAnchor?: number | null;
  lookahead?: { x: number; y: number } | null;
  /** Predicted horizon polyline in map frame (e.g. MPC). */
  pathOverlay?: { x: number; y: number }[] | null;
  /** Particle-filter pose in map frame (shown during race alongside odom). */
  amclPose?: { x: number; y: number; yaw: number } | null;
  mode: TrackMode;
  tool?: EditTool;
  brushRadius?: number;
  onStroke?: (points: [number, number][], tool: EditTool, radius: number) => void;
  onAnchorsChange?: (anchors: [number, number][], commit: boolean) => void;
  onAnchorSelect?: (index: number | null) => void;
  emptyText?: string;
  title: string;
  subtitle: string;
  meta?: string;
  legend?: { color: string; label: string; dashed?: boolean }[];
  /** Charts stacked under the map (legacy playground-style). */
  bottomExtra?: ReactNode;
  /** Charts in the right half; map stays left (Mapping race/raceline). */
  sideExtra?: ReactNode;
  /** Clear the odom path trail overlay. */
  onClearTrail?: () => void;
}

const COLORS = {
  car: "#0ea36f",
  amcl: "#7c3aed",
  trail: "rgba(59, 118, 232, 0.55)",
  anchor: "#ffffff",
  anchorBorder: "#1c2333",
  anchorSelected: "#3b76e8",
  preview: "#7c3aed",
  lookahead: "#d97706",
  pathOverlay: "#0891b2",
  violation: "#dc2626",
  eraseBrush: "rgba(14, 163, 111, 0.5)",
  wallBrush: "rgba(28, 35, 51, 0.6)",
};

function drawPoseArrow(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  yaw: number,
  fill: string,
  stroke?: string
) {
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(-yaw + Math.PI / 2);
  ctx.fillStyle = fill;
  ctx.beginPath();
  ctx.moveTo(0, -9);
  ctx.lineTo(5.5, 7);
  ctx.lineTo(-5.5, 7);
  ctx.closePath();
  ctx.fill();
  if (stroke) {
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }
  ctx.restore();
}

/** velocity → color (blue slow → red fast) */
function velColor(v: number, vmin: number, vmax: number): string {
  const t = Math.max(0, Math.min(1, (v - vmin) / Math.max(vmax - vmin, 0.01)));
  const hue = 220 - t * 220; // 220 (blue) → 0 (red)
  return `hsl(${hue}, 82%, 52%)`;
}

interface View {
  scale: number; // canvas px per meter
  cx: number; // world x at canvas center
  cy: number; // world y at canvas center
  fitted: string | null; // fitKey that was auto-fitted
}

export default function TrackCanvas({
  source,
  telemetry,
  trail,
  raceline,
  preview,
  anchors,
  selectedAnchor,
  lookahead,
  pathOverlay = null,
  amclPose = null,
  mode,
  tool = "none",
  brushRadius = 3,
  onStroke,
  onAnchorsChange,
  onAnchorSelect,
  emptyText = "Start mapping or select a saved map",
  title,
  subtitle,
  meta,
  legend,
  bottomExtra,
  sideExtra,
  onClearTrail,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  /** Offscreen map raster built from occupancy (putImageData). */
  const mapBmpRef = useRef<{
    key: string;
    canvas: HTMLCanvasElement | null;
    width: number;
    height: number;
  }>({
    key: "",
    canvas: null,
    width: 0,
    height: 0,
  });
  const viewRef = useRef<View>({ scale: 40, cx: 0, cy: 0, fitted: null });
  const stateRef = useRef({
    source,
    telemetry,
    trail,
    raceline,
    preview,
    anchors,
    selectedAnchor,
    lookahead,
    pathOverlay,
    amclPose,
    mode,
    tool,
    brushRadius,
  });
  stateRef.current = {
    source,
    telemetry,
    trail,
    raceline,
    preview,
    anchors,
    selectedAnchor,
    lookahead,
    pathOverlay,
    amclPose,
    mode,
    tool,
    brushRadius,
  };
  const cbRef = useRef({ onStroke, onAnchorsChange, onAnchorSelect });
  cbRef.current = { onStroke, onAnchorsChange, onAnchorSelect };

  // interaction state
  const dragRef = useRef<{
    kind: "none" | "pan" | "anchor" | "stroke";
    startX: number;
    startY: number;
    viewCx: number;
    viewCy: number;
    anchorIdx: number;
    strokeWorld: [number, number][];
  }>({
    kind: "none",
    startX: 0,
    startY: 0,
    viewCx: 0,
    viewCy: 0,
    anchorIdx: -1,
    strokeWorld: [],
  });
  const mouseRef = useRef<{ x: number; y: number; inside: boolean }>({
    x: 0,
    y: 0,
    inside: false,
  });

  // Identity of the raster we should be showing. MappingCanvas builds a new
  // `source` object every telemetry frame; depending on the object itself
  // aborted the grid fetch ~20 Hz and left the map (and therefore the
  // fitted lookahead / horizon overlays) blank until a lucky cached hit.
  const sourceKey = source?.key ?? "";
  const sourceGridUrl = source?.gridUrl ?? "";
  const sourceWidth = source?.width ?? 0;
  const sourceHeight = source?.height ?? 0;
  const sourceOcc = source?.occupancy;

  // --- occupancy → offscreen canvas -----------------------------------------
  useEffect(() => {
    if (!sourceKey) {
      mapBmpRef.current = { key: "", canvas: null, width: 0, height: 0 };
      return;
    }
    if (mapBmpRef.current.key === sourceKey && mapBmpRef.current.canvas) {
      return;
    }
    let cancelled = false;

    const apply = (occ: Int8Array, w: number, h: number) => {
      if (cancelled) return;
      if (occ.length < w * h) return;
      mapBmpRef.current = {
        key: sourceKey,
        canvas: occupancyToCanvas(occ, w, h),
        width: w,
        height: h,
      };
      // Keep fitted across key/version/liveSeq rebuilds — auto-fit only when
      // fitKey changes in the render loop, so pan/zoom is not fought.
    };

    if (sourceOcc && sourceOcc.length >= sourceWidth * sourceHeight) {
      apply(sourceOcc, sourceWidth, sourceHeight);
      return () => {
        cancelled = true;
      };
    }

    if (sourceGridUrl) {
      fetch(sourceGridUrl)
        .then((r) => {
          if (!r.ok) throw new Error(`grid ${r.status}`);
          return r.json();
        })
        .then((data: { occ_b64?: string; width?: number; height?: number }) => {
          if (cancelled || !data.occ_b64) return;
          const w = Number(data.width ?? sourceWidth);
          const h = Number(data.height ?? sourceHeight);
          apply(decodeOccB64(data.occ_b64), w, h);
        })
        .catch(() => {
          /* leave blank; effect re-runs if the URL/key changes */
        });
      return () => {
        cancelled = true;
      };
    }

    mapBmpRef.current = { key: "", canvas: null, width: 0, height: 0 };
    return () => {
      cancelled = true;
    };
  }, [sourceKey, sourceGridUrl, sourceWidth, sourceHeight, sourceOcc]);

  // --- coordinate helpers ---------------------------------------------------
  const worldToCanvas = useCallback(
    (wx: number, wy: number, w: number, h: number) => {
      const v = viewRef.current;
      return {
        x: w / 2 + (wx - v.cx) * v.scale,
        y: h / 2 - (wy - v.cy) * v.scale,
      };
    },
    []
  );

  const canvasToWorld = useCallback((px: number, py: number, w: number, h: number) => {
    const v = viewRef.current;
    return {
      x: v.cx + (px - w / 2) / v.scale,
      y: v.cy - (py - h / 2) / v.scale,
    };
  }, []);

  const worldToMapPixel = useCallback((wx: number, wy: number) => {
    const s = stateRef.current.source;
    if (!s) return null;
    return [
      (wx - s.origin[0]) / s.resolution,
      s.height - 1 - (wy - s.origin[1]) / s.resolution,
    ] as [number, number];
  }, []);

  /**
   * Convert a world-frame point (odom) into map-frame coordinates for overlay.
   *
   * Preferred: `mapToWorld.frame === "map_to_world"` (map → world):
   *   p_world = R(yaw) * p_map + t  ⇒  p_map = R(-yaw) * (p_world - t)
   *
   * Legacy saves (no frame tag) stored world → map under this key — apply
   * that transform directly.
   */
  const toMapFrame = useCallback((wx: number, wy: number): [number, number] => {
    const t = stateRef.current.source?.mapToWorld as
      | (Transform2D & { frame?: string })
      | null
      | undefined;
    if (!t) return [wx, wy];

    const c = Math.cos(t.yaw);
    const s = Math.sin(t.yaw);

    if (t.frame === "map_to_world") {
      const dx = wx - t.tx;
      const dy = wy - t.ty;
      return [c * dx + s * dy, -s * dx + c * dy];
    }

    // Legacy world → map (or untagged old saves)
    return [t.tx + c * wx - s * wy, t.ty + s * wx + c * wy];
  }, []);

  // --- render loop ------------------------------------------------------------
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    let raf = 0;

    const render = () => {
      raf = requestAnimationFrame(render);
      const dpr = window.devicePixelRatio || 1;
      const box = canvas.parentElement;
      const w = Math.floor(box?.clientWidth || 0);
      const h = Math.floor(box?.clientHeight || 0);
      if (w < 2 || h < 2) return;
      if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
        canvas.width = w * dpr;
        canvas.height = h * dpr;
      }
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);

      const st = stateRef.current;
      const src = st.source;
      const view = viewRef.current;

      // Auto-fit on new track identity (prefer loaded grid size over meta).
      if (src && view.fitted !== src.fitKey) {
        const hasBmp =
          mapBmpRef.current.key === src.key && mapBmpRef.current.width > 0;
        // Wait for grid fetch before fitting when only a URL is available.
        if (!hasBmp && src.gridUrl && !src.occupancy) {
          // keep waiting
        } else {
          const gw = hasBmp ? mapBmpRef.current.width : src.width;
          const gh = hasBmp ? mapBmpRef.current.height : src.height;
          const mw = gw * src.resolution;
          const mh = gh * src.resolution;
          if (mw > 0 && mh > 0) {
            view.scale = Math.min(w / mw, h / mh) * 0.92;
            view.cx = src.origin[0] + mw / 2;
            view.cy = src.origin[1] + mh / 2;
            view.fitted = src.fitKey;
          }
        }
      }

      // Map occupancy raster (crisp pixels — never smooth)
      const mapBmp = mapBmpRef.current.canvas;
      if (src && mapBmp && mapBmpRef.current.key === src.key) {
        const gw = mapBmpRef.current.width || src.width;
        const gh = mapBmpRef.current.height || src.height;
        const topLeft = worldToCanvas(
          src.origin[0],
          src.origin[1] + gh * src.resolution,
          w,
          h
        );
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(
          mapBmp,
          topLeft.x,
          topLeft.y,
          gw * src.resolution * view.scale,
          gh * src.resolution * view.scale
        );
        ctx.imageSmoothingEnabled = true;
      }

      // Optimizer preview (dashed)
      if (st.preview && st.preview.x.length > 1) {
        ctx.strokeStyle = COLORS.preview;
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 5]);
        ctx.beginPath();
        for (let i = 0; i < st.preview.x.length; i++) {
          const p = worldToCanvas(st.preview.x[i], st.preview.y[i], w, h);
          if (i === 0) ctx.moveTo(p.x, p.y);
          else ctx.lineTo(p.x, p.y);
        }
        ctx.closePath();
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Raceline (velocity-colored, clearance violations flagged)
      const rl = st.raceline;
      if (rl && rl.x.length > 1) {
        let vmin = Infinity;
        let vmax = -Infinity;
        for (const v of rl.v) {
          if (v < vmin) vmin = v;
          if (v > vmax) vmax = v;
        }
        const n = rl.x.length;
        // violation halo underneath
        if (rl.clearance && rl.margin != null) {
          ctx.lineWidth = 7;
          ctx.strokeStyle = "rgba(220, 38, 38, 0.35)";
          for (let i = 0; i < n; i++) {
            if (rl.clearance[i] >= rl.margin) continue;
            const j = (i + 1) % n;
            const a = worldToCanvas(rl.x[i], rl.y[i], w, h);
            const b = worldToCanvas(rl.x[j], rl.y[j], w, h);
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.stroke();
          }
        }
        ctx.lineWidth = 2.5;
        ctx.lineCap = "round";
        for (let i = 0; i < n; i++) {
          const j = (i + 1) % n;
          const a = worldToCanvas(rl.x[i], rl.y[i], w, h);
          const b = worldToCanvas(rl.x[j], rl.y[j], w, h);
          ctx.strokeStyle = velColor(rl.v[i], vmin, vmax);
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
        ctx.lineCap = "butt";
      }

      // Anchors
      if (st.mode === "edit-line" && st.anchors && st.anchors.length) {
        for (let i = 0; i < st.anchors.length; i++) {
          const p = worldToCanvas(st.anchors[i][0], st.anchors[i][1], w, h);
          const sel = i === st.selectedAnchor;
          ctx.beginPath();
          ctx.arc(p.x, p.y, sel ? 7 : 5, 0, Math.PI * 2);
          ctx.fillStyle = sel ? COLORS.anchorSelected : COLORS.anchor;
          ctx.fill();
          ctx.lineWidth = 1.5;
          ctx.strokeStyle = COLORS.anchorBorder;
          ctx.stroke();
        }
      }

      // Trail (world frame → map frame)
      if (st.trail.length > 1) {
        ctx.strokeStyle = COLORS.trail;
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        st.trail.forEach(([x, y], i) => {
          const m = toMapFrame(x, y);
          const p = worldToCanvas(m[0], m[1], w, h);
          if (i === 0) ctx.moveTo(p.x, p.y);
          else ctx.lineTo(p.x, p.y);
        });
        ctx.stroke();
      }

      // MPC predicted horizon (map frame)
      if (st.pathOverlay && st.pathOverlay.length > 1) {
        ctx.strokeStyle = COLORS.pathOverlay;
        ctx.lineWidth = 2.2;
        ctx.beginPath();
        st.pathOverlay.forEach((pt, i) => {
          const p = worldToCanvas(pt.x, pt.y, w, h);
          if (i === 0) ctx.moveTo(p.x, p.y);
          else ctx.lineTo(p.x, p.y);
        });
        ctx.stroke();
        const tip = st.pathOverlay[st.pathOverlay.length - 1];
        const tp = worldToCanvas(tip.x, tip.y, w, h);
        ctx.fillStyle = COLORS.pathOverlay;
        ctx.beginPath();
        ctx.arc(tp.x, tp.y, 3.5, 0, Math.PI * 2);
        ctx.fill();
      }

      // Lookahead / MPC ref target (map frame — same as raceline)
      if (st.lookahead) {
        const p = worldToCanvas(st.lookahead.x, st.lookahead.y, w, h);
        const carM = toMapFrame(
          st.telemetry.position[0],
          st.telemetry.position[1]
        );
        const cp = worldToCanvas(carM[0], carM[1], w, h);
        ctx.strokeStyle = COLORS.lookahead;
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(cp.x, cp.y);
        ctx.lineTo(p.x, p.y);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = COLORS.lookahead;
        ctx.beginPath();
        ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
        ctx.fill();
      }

      // Car (world-frame odom → map frame for overlay)
      {
        const [cxw, cyw] = st.telemetry.position;
        const m = toMapFrame(cxw, cyw);
        const p = worldToCanvas(m[0], m[1], w, h);
        const tf = src?.mapToWorld as
          | (Transform2D & { frame?: string })
          | null
          | undefined;
        // Heading: world yaw mapped into map frame.
        let yaw = st.telemetry.yaw;
        if (tf) {
          yaw =
            tf.frame === "map_to_world"
              ? st.telemetry.yaw - tf.yaw
              : st.telemetry.yaw + tf.yaw;
        }
        drawPoseArrow(ctx, p.x, p.y, yaw, COLORS.car);

        // Particle-filter pose is already in map frame (what Pure Pursuit uses).
        const amcl = st.amclPose;
        if (
          amcl &&
          Number.isFinite(amcl.x) &&
          Number.isFinite(amcl.y) &&
          Number.isFinite(amcl.yaw)
        ) {
          const ap = worldToCanvas(amcl.x, amcl.y, w, h);
          ctx.strokeStyle = "rgba(124, 58, 237, 0.55)";
          ctx.lineWidth = 1.4;
          ctx.setLineDash([3, 3]);
          ctx.beginPath();
          ctx.moveTo(p.x, p.y);
          ctx.lineTo(ap.x, ap.y);
          ctx.stroke();
          ctx.setLineDash([]);
          drawPoseArrow(ctx, ap.x, ap.y, amcl.yaw, COLORS.amcl);
        }
      }

      // In-flight brush stroke overlay
      const drag = dragRef.current;
      if (drag.kind === "stroke" && drag.strokeWorld.length > 0) {
        ctx.strokeStyle =
          st.tool === "erase" ? COLORS.eraseBrush : COLORS.wallBrush;
        ctx.lineWidth = Math.max(
          2,
          st.brushRadius * 2 * (src ? src.resolution : 0.05) * view.scale
        );
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.beginPath();
        drag.strokeWorld.forEach(([x, y], i) => {
          const p = worldToCanvas(x, y, w, h);
          if (i === 0) ctx.moveTo(p.x, p.y);
          else ctx.lineTo(p.x, p.y);
        });
        ctx.stroke();
        ctx.lineCap = "butt";
        ctx.lineJoin = "miter";
      }

      // Brush cursor
      if (
        st.mode === "edit-map" &&
        st.tool !== "none" &&
        mouseRef.current.inside
      ) {
        const r =
          st.brushRadius * (src ? src.resolution : 0.05) * view.scale;
        ctx.strokeStyle =
          st.tool === "erase" ? "#0ea36f" : "#1c2333";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(mouseRef.current.x, mouseRef.current.y, Math.max(r, 2), 0, Math.PI * 2);
        ctx.stroke();
      }

      // Scale bar (1 m)
      {
        const px = view.scale; // 1 m in px
        const x0 = 14;
        const y0 = h - 14;
        ctx.strokeStyle = "#667085";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(x0 + px, y0);
        ctx.stroke();
        ctx.fillStyle = "#667085";
        ctx.font = "10px ui-monospace, monospace";
        ctx.fillText("1 m", x0 + px / 2 - 8, y0 - 4);
      }
    };
    raf = requestAnimationFrame(render);
    return () => cancelAnimationFrame(raf);
  }, [worldToCanvas, toMapFrame]);

  // --- interactions -------------------------------------------------------------
  const hitAnchor = useCallback(
    (px: number, py: number, w: number, h: number): number => {
      const st = stateRef.current;
      if (!st.anchors) return -1;
      for (let i = 0; i < st.anchors.length; i++) {
        const p = worldToCanvas(st.anchors[i][0], st.anchors[i][1], w, h);
        if (Math.hypot(p.x - px, p.y - py) <= 9) return i;
      }
      return -1;
    },
    [worldToCanvas]
  );

  /** Nearest anchor-polygon segment within tolerance; returns insert index. */
  const hitSegment = useCallback(
    (px: number, py: number, w: number, h: number): number => {
      const st = stateRef.current;
      const a = st.anchors;
      if (!a || a.length < 2) return -1;
      let best = -1;
      let bestD = 9;
      for (let i = 0; i < a.length; i++) {
        const j = (i + 1) % a.length;
        const p1 = worldToCanvas(a[i][0], a[i][1], w, h);
        const p2 = worldToCanvas(a[j][0], a[j][1], w, h);
        const dx = p2.x - p1.x;
        const dy = p2.y - p1.y;
        const len2 = dx * dx + dy * dy;
        if (len2 < 1e-6) continue;
        const t = Math.max(
          0,
          Math.min(1, ((px - p1.x) * dx + (py - p1.y) * dy) / len2)
        );
        const d = Math.hypot(p1.x + t * dx - px, p1.y + t * dy - py);
        if (d < bestD) {
          bestD = d;
          best = i;
        }
      }
      return best;
    },
    [worldToCanvas]
  );

  const onPointerDown = useCallback(
    (e: React.PointerEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      canvas.setPointerCapture(e.pointerId);
      const rect = canvas.getBoundingClientRect();
      const px = e.clientX - rect.left;
      const py = e.clientY - rect.top;
      const w = rect.width;
      const h = rect.height;
      const st = stateRef.current;
      const drag = dragRef.current;
      drag.startX = px;
      drag.startY = py;
      drag.viewCx = viewRef.current.cx;
      drag.viewCy = viewRef.current.cy;

      const panButtons = e.button === 1 || e.shiftKey;

      if (e.button === 2) {
        // Right-click: delete anchor in edit-line mode (handled in ctx menu).
        return;
      }

      if (st.mode === "edit-map" && st.tool !== "none" && !panButtons) {
        const wpt = canvasToWorld(px, py, w, h);
        drag.kind = "stroke";
        drag.strokeWorld = [[wpt.x, wpt.y]];
        return;
      }

      if (st.mode === "edit-line" && !panButtons) {
        const ai = hitAnchor(px, py, w, h);
        if (ai >= 0) {
          drag.kind = "anchor";
          drag.anchorIdx = ai;
          cbRef.current.onAnchorSelect?.(ai);
          return;
        }
        const seg = hitSegment(px, py, w, h);
        if (seg >= 0 && st.anchors) {
          // Insert a new anchor on the clicked segment and start dragging it.
          const wpt = canvasToWorld(px, py, w, h);
          const next = [...st.anchors];
          next.splice(seg + 1, 0, [wpt.x, wpt.y]);
          cbRef.current.onAnchorsChange?.(next, false);
          drag.kind = "anchor";
          drag.anchorIdx = seg + 1;
          cbRef.current.onAnchorSelect?.(seg + 1);
          return;
        }
        cbRef.current.onAnchorSelect?.(null);
      }

      drag.kind = "pan";
    },
    [canvasToWorld, hitAnchor, hitSegment]
  );

  const onPointerMove = useCallback(
    (e: React.PointerEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const px = e.clientX - rect.left;
      const py = e.clientY - rect.top;
      const w = rect.width;
      const h = rect.height;
      mouseRef.current = { x: px, y: py, inside: true };
      const drag = dragRef.current;
      const st = stateRef.current;

      if (drag.kind === "pan") {
        const v = viewRef.current;
        v.cx = drag.viewCx - (px - drag.startX) / v.scale;
        v.cy = drag.viewCy + (py - drag.startY) / v.scale;
      } else if (drag.kind === "stroke") {
        const wpt = canvasToWorld(px, py, w, h);
        const last = drag.strokeWorld[drag.strokeWorld.length - 1];
        if (
          !last ||
          Math.hypot(wpt.x - last[0], wpt.y - last[1]) >
            (st.source?.resolution ?? 0.05)
        ) {
          drag.strokeWorld.push([wpt.x, wpt.y]);
        }
      } else if (drag.kind === "anchor" && st.anchors) {
        const wpt = canvasToWorld(px, py, w, h);
        const next = [...st.anchors];
        if (drag.anchorIdx >= 0 && drag.anchorIdx < next.length) {
          next[drag.anchorIdx] = [wpt.x, wpt.y];
          cbRef.current.onAnchorsChange?.(next, false);
        }
      }
    },
    [canvasToWorld]
  );

  const finishDrag = useCallback(() => {
    const drag = dragRef.current;
    const st = stateRef.current;
    if (drag.kind === "stroke" && drag.strokeWorld.length > 0) {
      const pts: [number, number][] = [];
      for (const [x, y] of drag.strokeWorld) {
        const p = worldToMapPixel(x, y);
        if (p) pts.push(p);
      }
      if (pts.length && st.tool !== "none") {
        cbRef.current.onStroke?.(pts, st.tool, st.brushRadius);
      }
    } else if (drag.kind === "anchor" && st.anchors) {
      cbRef.current.onAnchorsChange?.([...st.anchors], true);
    }
    drag.kind = "none";
    drag.strokeWorld = [];
    drag.anchorIdx = -1;
  }, [worldToMapPixel]);

  const onPointerUp = useCallback(
    (e: React.PointerEvent<HTMLCanvasElement>) => {
      canvasRef.current?.releasePointerCapture(e.pointerId);
      finishDrag();
    },
    [finishDrag]
  );

  // Native non-passive wheel listener — React's onWheel is often passive,
  // so preventDefault would be ignored and the page would scroll.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const px = e.clientX - rect.left;
      const py = e.clientY - rect.top;
      const w = rect.width;
      const h = rect.height;
      const v = viewRef.current;
      const before = {
        x: v.cx + (px - w / 2) / v.scale,
        y: v.cy - (py - h / 2) / v.scale,
      };
      const factor = Math.exp(-e.deltaY * 0.0012);
      v.scale = Math.max(2, Math.min(500, v.scale * factor));
      // keep the point under the cursor fixed
      v.cx = before.x - (px - w / 2) / v.scale;
      v.cy = before.y + (py - h / 2) / v.scale;
    };
    canvas.addEventListener("wheel", onWheel, { passive: false });
    return () => canvas.removeEventListener("wheel", onWheel);
  }, []);

  const onContextMenu = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      e.preventDefault();
      const st = stateRef.current;
      if (st.mode !== "edit-line" || !st.anchors) return;
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const ai = hitAnchor(
        e.clientX - rect.left,
        e.clientY - rect.top,
        rect.width,
        rect.height
      );
      if (ai >= 0 && st.anchors.length > 4) {
        const next = st.anchors.filter((_, i) => i !== ai);
        cbRef.current.onAnchorSelect?.(null);
        cbRef.current.onAnchorsChange?.(next, true);
      }
    },
    [hitAnchor]
  );

  // Delete key removes the selected anchor.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const st = stateRef.current;
      if (st.mode !== "edit-line") return;
      if (e.key !== "Delete" && e.key !== "Backspace") return;
      const target = e.target as HTMLElement | null;
      if (target && ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName))
        return;
      if (
        st.selectedAnchor != null &&
        st.anchors &&
        st.anchors.length > 4 &&
        st.selectedAnchor < st.anchors.length
      ) {
        e.preventDefault();
        const next = st.anchors.filter((_, i) => i !== st.selectedAnchor);
        cbRef.current.onAnchorSelect?.(null);
        cbRef.current.onAnchorsChange?.(next, true);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const cursor =
    mode === "edit-map" && tool !== "none"
      ? "crosshair"
      : mode === "edit-line"
        ? "default"
        : "grab";

  const boardClass = sideExtra
    ? "viz-board split-side"
    : bottomExtra
      ? "viz-board"
      : "viz-board single";

  return (
    <div className={boardClass}>
      <div className="viz-card viz-track">
        <div className="viz-title">
          {title} <span>{subtitle}</span>
          {meta && <span className="viz-meta">{meta}</span>}
          {onClearTrail && (
            <button
              type="button"
              className="btn small"
              disabled={trail.length === 0}
              onClick={onClearTrail}
            >
              Clear trail
            </button>
          )}
        </div>
        <div className="canvas-wrap">
          <canvas
            ref={canvasRef}
            className="viz"
            style={{ cursor, touchAction: "none" }}
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
            onPointerLeave={() => {
              mouseRef.current.inside = false;
            }}
            onContextMenu={onContextMenu}
          />
          {!source && <div className="canvas-placeholder">{emptyText}</div>}
        </div>
        {legend && legend.length > 0 && (
          <div className="viz-legend">
            {legend.map((item) => (
              <span className="legend-item" key={item.label}>
                <span
                  className={`legend-swatch${item.dashed ? " dashed" : ""}`}
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
        )}
      </div>
      {sideExtra && <div className="viz-side">{sideExtra}</div>}
      {!sideExtra && bottomExtra && (
        <div className="viz-bottom auto">{bottomExtra}</div>
      )}
    </div>
  );
}
