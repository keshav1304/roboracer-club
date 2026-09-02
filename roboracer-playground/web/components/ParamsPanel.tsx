"use client";

import { useEffect, useRef, useState } from "react";
import type { ParamsAck } from "@/lib/useGateway";
import type { AlgorithmSpec } from "@/lib/algorithms";

interface Props {
  algorithm: AlgorithmSpec;
  onSetParams: (params: Record<string, number>) => void;
  lastAck: ParamsAck | null;
  disabled: boolean;
}

export default function ParamsPanel({
  algorithm,
  onSetParams,
  lastAck,
  disabled,
}: Props) {
  const [values, setValues] = useState<Record<string, number>>(() =>
    Object.fromEntries(algorithm.params.map((s) => [s.name, s.default]))
  );
  const pending = useRef<Record<string, number>>({});
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Reset slider values when the algorithm changes.
  useEffect(() => {
    setValues(
      Object.fromEntries(algorithm.params.map((s) => [s.name, s.default]))
    );
    pending.current = {};
  }, [algorithm]);

  // Debounced live send (~80ms) so sliders feel real-time without spamming.
  const queueSend = (name: string, value: number) => {
    pending.current[name] = value;
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => {
      onSetParams({ ...pending.current });
      pending.current = {};
    }, 80);
  };

  useEffect(
    () => () => {
      if (timer.current) clearTimeout(timer.current);
    },
    []
  );

  return (
    <div>
      <div className="field-label">Parameters</div>
      {algorithm.params.map((s) => (
        <div className="slider-group" key={s.name}>
          <div className="slider-label">
            <span className="slider-name">
              {s.label}
              <span className="info-tip" tabIndex={0} aria-label={s.info}>
                <span className="info-icon" aria-hidden>
                  i
                </span>
                <span className="info-bubble" role="tooltip">
                  {s.info}
                </span>
              </span>
            </span>
            <span className="val">
              {(values[s.name] ?? s.default).toFixed(s.step < 0.01 ? 3 : 2)}
            </span>
          </div>
          <input
            type="range"
            min={s.min}
            max={s.max}
            step={s.step}
            value={values[s.name] ?? s.default}
            disabled={disabled}
            onChange={(e) => {
              const v = parseFloat(e.target.value);
              setValues((prev) => ({ ...prev, [s.name]: v }));
              queueSend(s.name, v);
            }}
          />
        </div>
      ))}
      {lastAck && !lastAck.ok && (
        <div className="ack fail">Rejected: {lastAck.reason}</div>
      )}
      {disabled && (
        <div className="ack fail">Start the algorithm to enable tuning.</div>
      )}
    </div>
  );
}
