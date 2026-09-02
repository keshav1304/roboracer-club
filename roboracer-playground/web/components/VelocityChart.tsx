"use client";

import { useEffect, useRef } from "react";

interface Props {
  s: number[];
  v: number[];
  /** Car's Frenet s from pure pursuit (particle-filter pose), if racing. */
  carS?: number | null;
  /** True simulator speed from odom twist (m/s) — not algo_debug. */
  carSpeed?: number | null;
  statsText?: string;
}

function velColor(v: number, vmin: number, vmax: number): string {
  const t = Math.max(0, Math.min(1, (v - vmin) / Math.max(vmax - vmin, 0.01)));
  const hue = 220 - t * 220;
  return `hsl(${hue}, 82%, 52%)`;
}

/** Velocity profile v(s) along the raceline. */
export default function VelocityChart({ s, v, carS, carSpeed, statsText }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const propsRef = useRef({ s, v, carS, carSpeed });
  propsRef.current = { s, v, carS, carSpeed };

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

      const { s, v, carS, carSpeed } = propsRef.current;
      ctx.font = "10px ui-monospace, monospace";
      if (s.length < 2 || v.length < 2) {
        ctx.fillStyle = "#98a2b3";
        ctx.textAlign = "center";
        ctx.fillText("Generate or edit a raceline to see v(s)", w / 2, h / 2);
        ctx.textAlign = "start";
        return;
      }

      const pad = { l: 30, r: 8, t: 8, b: 16 };
      const maxS = s[s.length - 1] || 1;
      let vmin = Infinity;
      let vmax = -Infinity;
      for (const val of v) {
        if (val < vmin) vmin = val;
        if (val > vmax) vmax = val;
      }
      const span = Math.max(vmax - vmin, 0.5);
      const y0 = Math.max(0, vmin - span * 0.1);
      const y1 = vmax + span * 0.1;

      const X = (si: number) => pad.l + (si / maxS) * (w - pad.l - pad.r);
      const Y = (vi: number) =>
        h - pad.b - ((vi - y0) / (y1 - y0)) * (h - pad.t - pad.b);

      // Axis labels
      ctx.fillStyle = "#98a2b3";
      ctx.fillText(`${y1.toFixed(1)}`, 2, pad.t + 8);
      ctx.fillText(`${y0.toFixed(1)} m/s`, 2, h - pad.b);
      ctx.fillText(`${maxS.toFixed(0)} m`, w - pad.r - 34, h - 3);

      // Grid line at mid velocity
      ctx.strokeStyle = "#dfe5ef";
      ctx.lineWidth = 1;
      const midV = (y0 + y1) / 2;
      ctx.beginPath();
      ctx.moveTo(pad.l, Y(midV));
      ctx.lineTo(w - pad.r, Y(midV));
      ctx.stroke();

      // Colored profile
      ctx.lineWidth = 2;
      for (let i = 0; i < s.length - 1; i++) {
        ctx.strokeStyle = velColor(v[i], vmin, vmax);
        ctx.beginPath();
        ctx.moveTo(X(s[i]), Y(v[i]));
        ctx.lineTo(X(s[i + 1]), Y(v[i + 1]));
        ctx.stroke();
      }

      // Car position marker
      if (carS != null && Number.isFinite(carS)) {
        const x = X(Math.max(0, Math.min(maxS, carS)));
        ctx.strokeStyle = "#0ea36f";
        ctx.lineWidth = 1.5;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(x, pad.t);
        ctx.lineTo(x, h - pad.b);
        ctx.stroke();
        ctx.setLineDash([]);
        if (carSpeed != null && Number.isFinite(carSpeed)) {
          ctx.fillStyle = "#0ea36f";
          ctx.beginPath();
          ctx.arc(x, Y(Math.max(y0, Math.min(y1, carSpeed))), 4, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    };
    raf = requestAnimationFrame(render);
    return () => cancelAnimationFrame(raf);
  }, []);

  return (
    <div className="viz-card">
      <div className="viz-title">
        Velocity profile <span>v(s) · green = true odom speed</span>
        {statsText && <span className="viz-meta">{statsText}</span>}
      </div>
      <div className="canvas-wrap">
        <canvas ref={canvasRef} className="viz" />
      </div>
    </div>
  );
}
