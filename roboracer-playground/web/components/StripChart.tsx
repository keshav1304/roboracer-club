"use client";

import { useEffect, useRef } from "react";

interface Props {
  /** Samples, newest last. */
  values: number[];
  label: string;
  subtitle: string;
  unit: string;
  /** |value| above this renders the readout in amber. */
  warnAbs: number;
  /** Minimum half-range of the y axis so tiny noise isn't over-zoomed. */
  minRange: number;
  decimals?: number;
  emptyText?: string;
}

/** Scrolling zero-centered strip chart for a single debug signal. */
export default function StripChart({
  values,
  label,
  subtitle,
  unit,
  warnAbs,
  minRange,
  decimals = 2,
  emptyText = "Start the algorithm to chart data",
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const propsRef = useRef({ values, unit, warnAbs, minRange, decimals, emptyText });
  propsRef.current = { values, unit, warnAbs, minRange, decimals, emptyText };

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

      const { values, unit, warnAbs, minRange, decimals, emptyText } =
        propsRef.current;

      ctx.fillStyle = "#98a2b3";
      ctx.font = "11px ui-monospace, monospace";
      if (values.length < 2) {
        ctx.textAlign = "center";
        ctx.fillText(emptyText, w / 2, h / 2);
        ctx.textAlign = "start";
        return;
      }

      const pad = 8;
      const midY = h / 2;
      let maxAbs = minRange;
      for (const v of values) maxAbs = Math.max(maxAbs, Math.abs(v));
      const scaleY = (h / 2 - pad) / maxAbs;

      // Zero line
      ctx.strokeStyle = "#dde2ec";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, midY);
      ctx.lineTo(w, midY);
      ctx.stroke();

      // Range labels
      ctx.fillStyle = "#98a2b3";
      ctx.fillText(`+${maxAbs.toFixed(decimals)} ${unit}`, 4, pad + 10);
      ctx.fillText(`-${maxAbs.toFixed(decimals)} ${unit}`, 4, h - 4);

      // Polyline
      const dx = (w - pad * 2) / Math.max(values.length - 1, 1);
      ctx.strokeStyle = "#3b76e8";
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      values.forEach((v, i) => {
        const x = pad + i * dx;
        const y = midY - v * scaleY;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();

      // Latest value readout
      const last = values[values.length - 1];
      ctx.fillStyle = Math.abs(last) > warnAbs ? "#d97706" : "#0ea36f";
      ctx.textAlign = "right";
      ctx.fillText(`${last.toFixed(decimals + 1)} ${unit}`, w - 6, pad + 10);
      ctx.textAlign = "start";
    };

    raf = requestAnimationFrame(render);
    return () => cancelAnimationFrame(raf);
  }, []);

  return (
    <div className="viz-card">
      <div className="viz-title">
        {label} <span>{subtitle}</span>
        <span className="viz-meta">
          {values.length > 1 ? `${values.length} samples` : "No data"}
        </span>
      </div>
      <div className="canvas-wrap">
        <canvas ref={canvasRef} className="viz" />
      </div>
    </div>
  );
}
