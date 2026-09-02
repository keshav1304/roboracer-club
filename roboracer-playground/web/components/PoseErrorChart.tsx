"use client";

import { useEffect, useRef } from "react";

interface Props {
  /** Planar error ‖p_true − p_pf‖ in map frame (meters). */
  xy: number[];
  /** Yaw error true − particle filter (degrees), signed shortest angle. */
  yawDeg: number[];
  /** Latest readout override (shown in title meta). */
  latestXy?: number | null;
  latestYawDeg?: number | null;
}

/** Dual strip: true odom pose vs particle-filter pose (map frame). */
export default function PoseErrorChart({
  xy,
  yawDeg,
  latestXy,
  latestYawDeg,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const propsRef = useRef({ xy, yawDeg });
  propsRef.current = { xy, yawDeg };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    let raf = 0;

    const render = () => {
      raf = requestAnimationFrame(render);
      const dpr = window.devicePixelRatio || 1;
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      if (w === 0 || h === 0) return;
      if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
        canvas.width = w * dpr;
        canvas.height = h * dpr;
      }
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);

      const { xy, yawDeg } = propsRef.current;
      ctx.font = "10px ui-monospace, monospace";
      if (xy.length < 2) {
        ctx.fillStyle = "#98a2b3";
        ctx.textAlign = "center";
        ctx.fillText(
          "Start controller to chart true odom versus particle filter",
          w / 2,
          h / 2
        );
        ctx.textAlign = "start";
        return;
      }

      const mid = Math.floor(h / 2);
      const pad = 6;
      drawStrip(ctx, xy, 0, mid, w, pad, "#3b76e8", "m", 0.05, 2);
      drawStrip(ctx, yawDeg, mid, h - mid, w, pad, "#d97706", "°", 2, 1);

      // Divider
      ctx.strokeStyle = "#e6ebf3";
      ctx.beginPath();
      ctx.moveTo(0, mid);
      ctx.lineTo(w, mid);
      ctx.stroke();

      ctx.fillStyle = "#98a2b3";
      ctx.fillText("xy err", 4, pad + 10);
      ctx.fillText("yaw err", 4, mid + pad + 10);
    };
    raf = requestAnimationFrame(render);
    return () => cancelAnimationFrame(raf);
  }, []);

  const xyRead =
    latestXy != null && Number.isFinite(latestXy)
      ? latestXy
      : xy.length
        ? xy[xy.length - 1]
        : null;
  const yawRead =
    latestYawDeg != null && Number.isFinite(latestYawDeg)
      ? latestYawDeg
      : yawDeg.length
        ? yawDeg[yawDeg.length - 1]
        : null;
  const meta =
    xyRead != null && yawRead != null
      ? `${xyRead.toFixed(3)} m · ${yawRead.toFixed(1)}°`
      : undefined;

  return (
    <div className="viz-card">
      <div className="viz-title">
        Pose error <span>True odom versus Particle Filter</span>
        {meta && <span className="viz-meta">{meta}</span>}
      </div>
      <div className="canvas-wrap">
        <canvas ref={canvasRef} className="viz" />
      </div>
    </div>
  );
}

function drawStrip(
  ctx: CanvasRenderingContext2D,
  values: number[],
  y0: number,
  height: number,
  width: number,
  pad: number,
  color: string,
  unit: string,
  minRange: number,
  decimals: number
) {
  const midY = y0 + height / 2;
  let maxAbs = minRange;
  for (const v of values) maxAbs = Math.max(maxAbs, Math.abs(v));
  const scaleY = (height / 2 - pad) / maxAbs;

  ctx.strokeStyle = "#dde2ec";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, midY);
  ctx.lineTo(width, midY);
  ctx.stroke();

  ctx.fillStyle = "#98a2b3";
  ctx.fillText(`±${maxAbs.toFixed(decimals)}${unit}`, width - 52, y0 + pad + 8);

  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  const n = values.length;
  for (let i = 0; i < n; i++) {
    const x = (i / Math.max(n - 1, 1)) * width;
    const y = midY - values[i] * scaleY;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
}
