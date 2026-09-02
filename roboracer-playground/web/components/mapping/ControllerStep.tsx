"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { Gateway } from "@/lib/useGateway";
import type { Mapping } from "@/lib/useMapping";
import {
  algorithmSpec,
  algorithmsInGroup,
  isMapControllerAlgo,
} from "@/lib/algorithms";
import ParamsPanel from "@/components/ParamsPanel";

interface Props {
  gw: Gateway;
  lab: Mapping;
}

export default function ControllerStep({ gw, lab }: Props) {
  const { status, telemetry } = gw;
  const controllers = useMemo(() => algorithmsInGroup("map"), []);
  const [controller, setController] = useState(
    () => controllers[0]?.id ?? "pure_pursuit"
  );
  const racing = isMapControllerAlgo(status.algorithm);
  const activeController = racing ? status.algorithm! : controller;
  const spec = algorithmSpec(activeController);

  const [laps, setLaps] = useState<number[]>([]);
  const prevLap = useRef({ count: 0, time: 0 });
  const [startError, setStartError] = useState<string | null>(null);
  const lastAckN = useRef(0);

  // Record completed lap times from the lap counter edges. A jump of
  // exactly +1 is a real lap; anything else is a session reset.
  useEffect(() => {
    const { lap_count, lap_time } = telemetry;
    const prev = prevLap.current;
    if (lap_count === prev.count + 1 && prev.time > 0) {
      setLaps((l) => [...l, prev.time].slice(-30));
    }
    prevLap.current = { count: lap_count, time: lap_time };
  }, [telemetry]);

  // Surface start failures (e.g. missing raceline).
  useEffect(() => {
    const ack = gw.algoAck;
    if (!ack || ack.n === lastAckN.current) return;
    lastAckN.current = ack.n;
    setStartError(ack.ok ? null : ack.reason ?? "Failed to start");
  }, [gw.algoAck]);

  const canStart =
    status.wsConnected &&
    status.simConnected &&
    !status.slam.active &&
    lab.selectedRaceline != null &&
    !racing;

  return (
    <div className="step-body">
      <section className="step-section">
        <div className="field-label">Raceline</div>
        <select
          className="select"
          value={lab.selectedRaceline ?? ""}
          disabled={racing}
          onChange={(e) => lab.setSelectedRaceline(e.target.value || null)}
        >
          <option value="">Select a raceline…</option>
          {lab.racelines.map((r) => (
            <option key={r.name} value={r.name}>
              {r.name}
              {r.stats ? ` (~${r.stats.lap_time_est.toFixed(1)} s)` : ""}
            </option>
          ))}
        </select>
        {lab.racelines.length === 0 && (
          <div className="note-inline">
            No racelines yet. Create one in Generate Raceline.
          </div>
        )}

        <div className="field-label">Controller</div>
        <select
          className="select"
          value={activeController}
          disabled={racing}
          onChange={(e) => setController(e.target.value)}
        >
          {controllers.map((c) => (
            <option key={c.id} value={c.id}>
              {c.label}
            </option>
          ))}
        </select>

        <div className="btn-row">
          {!racing ? (
            <button
              className="btn primary"
              disabled={!canStart}
              onClick={() => {
                setLaps([]);
                prevLap.current = { count: telemetry.lap_count, time: 0 };
                setStartError(null);
                gw.startAlgo(controller, lab.selectedRaceline!);
              }}
            >
              Start
            </button>
          ) : (
            <button className="btn danger" onClick={() => gw.stopAlgo()}>
              Stop
            </button>
          )}
          <button
            className="btn"
            disabled={!status.wsConnected || !status.simConnected}
            onClick={() => gw.reset()}
          >
            Reset car
          </button>
        </div>
        {startError && <div className="ack fail">{startError}</div>}
        {status.slam.active && (
          <div className="note-inline">
            Stop mapping in Build Map before racing.
          </div>
        )}
        {racing && (
          <div className="note-inline">
            Particle filter:{" "}
            {status.localize.localized
              ? "localized"
              : status.localize.pose_fresh
                ? "tracking (high covariance)"
                : status.localize.active
                  ? "waiting for pose"
                  : "inactive"}
            {status.localize.map ? ` · map ${status.localize.map}` : ""}
          </div>
        )}
        {!racing && (
          <div className="note-inline">
            Set the simulator to Autonomous before starting. Control uses the
            particle filter on the raceline map; the canvas shows ground-truth
            odometry.
          </div>
        )}
      </section>

      <section className="step-section">
        <ParamsPanel
          algorithm={spec}
          onSetParams={gw.setParams}
          lastAck={gw.lastAck}
          disabled={!racing || !status.wsConnected}
        />
      </section>

      <section className="step-section">
        <div className="field-label">Laps</div>
        <div className="lap-table">
          <div className="lap-row lap-head">
            <span>#</span>
            <span>Time</span>
            <span></span>
          </div>
          {laps.length === 0 && (
            <div className="note-inline">Lap times show up here.</div>
          )}
          {laps.map((t, i) => {
            const best = Math.min(...laps);
            return (
              <div className="lap-row" key={i}>
                <span>{i + 1}</span>
                <span className={t === best ? "v good" : "v"}>
                  {t.toFixed(2)} s
                </span>
                <span>{t === best ? "best" : ""}</span>
              </div>
            );
          })}
          {racing && (
            <div className="lap-row">
              <span>{telemetry.lap_count + 1}</span>
              <span className="v">{telemetry.lap_time.toFixed(2)} s</span>
              <span>running</span>
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
