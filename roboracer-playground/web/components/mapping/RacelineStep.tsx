"use client";

import { useEffect, useRef, useState } from "react";
import type { Gateway } from "@/lib/useGateway";
import type { Mapping } from "@/lib/useMapping";
import NameModal from "./NameModal";
import SliderRow from "./SliderRow";

interface Props {
  gw: Gateway;
  lab: Mapping;
}

export default function RacelineStep({ gw, lab }: Props) {
  const [showSave, setShowSave] = useState(false);
  const [savePending, setSavePending] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const lastAckN = useRef(0);
  const pendingSaveName = useRef<string | null>(null);

  // Mirror map save: close modal, select the artifact, open Controller.
  useEffect(() => {
    const ack = gw.racelineAck;
    if (!ack || ack.op !== "save" || ack.n === lastAckN.current) return;
    lastAckN.current = ack.n;
    setSavePending(false);
    if (ack.ok) {
      setShowSave(false);
      setSaveError(null);
      const saved = (ack as { raceline?: { name?: string } }).raceline;
      const name = saved?.name ?? pendingSaveName.current;
      pendingSaveName.current = null;
      if (name) {
        lab.setSelectedRaceline(name);
        lab.setStep("controller");
      }
    } else {
      setSaveError(ack.reason ?? "Save failed");
    }
  }, [gw.racelineAck, lab]);

  const opt = lab.opt;
  const rl = lab.raceline;
  const pct =
    opt.budget > 0 ? Math.min(100, (opt.evals / opt.budget) * 100) : 0;

  return (
    <div className="step-body">
      <section className="step-section">
        <div className="field-label">Map</div>
        <select
          className="select"
          value={lab.editorMap ?? ""}
          disabled={opt.running}
          onChange={(e) => lab.setEditorMap(e.target.value || null)}
        >
          <option value="">Select a map…</option>
          {lab.maps.map((m) => (
            <option key={m.name} value={m.name}>
              {m.name}
            </option>
          ))}
        </select>
        {lab.maps.length === 0 && (
          <div className="note-inline">
            No maps yet. Build one in Build Map first.
          </div>
        )}
      </section>

      <section className="step-section">
        <div className="field-label">Optimize</div>
        <SliderRow
          label="Wall margin (m)"
          info="Minimum clearance from walls."
          min={0.08}
          max={0.5}
          step={0.01}
          value={lab.optParams.margin}
          disabled={opt.running}
          onChange={(v) => lab.setOptParams((p) => ({ ...p, margin: v }))}
        />
        <SliderRow
          label="Budget (evaluations)"
          info="More evaluations usually give a better line, but take longer."
          min={1000}
          max={30000}
          step={1000}
          value={lab.optParams.budget}
          disabled={opt.running}
          onChange={(v) => lab.setOptParams((p) => ({ ...p, budget: v }))}
        />
        <SliderRow
          label="Control points"
          info="How many lateral degrees of freedom along the centerline."
          min={30}
          max={120}
          step={5}
          value={lab.optParams.n_ctrl}
          disabled={opt.running}
          onChange={(v) => lab.setOptParams((p) => ({ ...p, n_ctrl: v }))}
        />
        {!opt.running ? (
          <button
            className="btn primary"
            style={{ width: "100%" }}
            disabled={!lab.editorMap || !gw.status.wsConnected}
            onClick={lab.startOptimize}
          >
            Generate raceline
          </button>
        ) : (
          <>
            <div className="progress-wrap">
              <div className="progress-bar" style={{ width: `${pct}%` }} />
            </div>
            <div className="progress-text">
              {opt.evals.toLocaleString()} / {opt.budget.toLocaleString()} evals
              {opt.bestCost != null && <> · cost {opt.bestCost.toFixed(3)}</>}
            </div>
            <button
              className="btn danger"
              style={{ width: "100%", marginTop: 8 }}
              onClick={lab.cancelOptimize}
            >
              Cancel
            </button>
          </>
        )}
        {opt.error && <div className="ack fail">{opt.error}</div>}
      </section>

      <section className="step-section">
        <div className="field-label">Edit and speed</div>
        {rl ? (
          <div className="stats-grid">
            <div>
              <span className="k">Length</span>
              <span className="v">{rl.length_m.toFixed(1)} m</span>
            </div>
            <div>
              <span className="k">Est. lap</span>
              <span className="v">{rl.lap_time_est.toFixed(1)} s</span>
            </div>
            <div>
              <span className="k">Clearance</span>
              <span
                className={`v ${
                  rl.min_clearance_m != null && rl.min_clearance_m < rl.margin
                    ? "bad"
                    : "good"
                }`}
              >
                {rl.min_clearance_m != null
                  ? `${rl.min_clearance_m.toFixed(2)} m`
                  : "n/a"}
              </span>
            </div>
            <div>
              <span className="k">Speed</span>
              <span className="v">
                {Math.min(...rl.v).toFixed(1)}–
                {Math.max(...rl.v).toFixed(1)} m/s
              </span>
            </div>
          </div>
        ) : (
          <div className="note-inline">
            Generate a line above or load one below. Drag white anchors on the
            map to edit. Click the line to add an anchor; right-click or Delete
            to remove one.
          </div>
        )}
        <SliderRow
          label="Max speed (m/s)"
          min={1}
          max={10}
          step={0.25}
          value={lab.fastParams.vmax}
          onChange={(v) => lab.setFastParam("vmax", v)}
        />
        <SliderRow
          label="Min speed (m/s)"
          min={0.2}
          max={3}
          step={0.1}
          value={lab.fastParams.vmin}
          onChange={(v) => lab.setFastParam("vmin", v)}
        />
        <SliderRow
          label="Lateral accel (m/s²)"
          info="Cornering limit for the velocity profile."
          min={1}
          max={8}
          step={0.25}
          value={lab.fastParams.alat}
          onChange={(v) => lab.setFastParam("alat", v)}
        />
        <SliderRow
          label="Spline smoothing"
          info="0 follows anchors exactly. Higher values smooth kinks."
          min={0}
          max={20}
          step={0.5}
          value={lab.fastParams.smooth}
          onChange={(v) => lab.setFastParam("smooth", v)}
        />
        <button
          className="btn primary"
          style={{ width: "100%" }}
          disabled={!rl || !lab.editorMap}
          onClick={() => {
            setSaveError(null);
            setShowSave(true);
          }}
        >
          Save raceline
        </button>
      </section>

      <section className="step-section">
        <div className="field-label">Saved racelines</div>
        {lab.racelines.length === 0 ? (
          <div className="note-inline">None saved yet.</div>
        ) : (
          <div className="artifact-list">
            {lab.racelines.map((r) => (
              <div
                key={r.name}
                className={`artifact ${lab.selectedRaceline === r.name ? "active" : ""}`}
              >
                <div className="artifact-info">
                  <div className="artifact-name">{r.name}</div>
                  <div className="artifact-sub">
                    {r.map ? `map ${r.map} · ` : ""}
                    {r.stats
                      ? `${r.stats.length_m.toFixed(0)} m · ~${r.stats.lap_time_est.toFixed(1)} s`
                      : ""}
                  </div>
                </div>
                <div className="artifact-actions">
                  <button
                    className="btn small"
                    onClick={() => lab.loadRacelineIntoEditor(r)}
                  >
                    Edit
                  </button>
                  <button
                    className="btn small"
                    onClick={() => {
                      lab.setSelectedRaceline(r.name);
                      lab.setStep("controller");
                    }}
                  >
                    Controller
                  </button>
                  <button
                    className="btn small danger"
                    onClick={() => {
                      if (confirm(`Delete raceline "${r.name}"?`)) {
                        gw.racelineDelete(r.name);
                      }
                    }}
                  >
                    ✕
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>

      {showSave && lab.editorMap && rl && (
        <NameModal
          title="Save raceline"
          description={`Tied to map “${lab.editorMap}”. Pure Pursuit will follow this line.`}
          suggestion={`${lab.editorMap}-line`}
          existingNames={lab.racelines.map((r) => r.name)}
          busy={savePending}
          error={saveError}
          onSave={(name) => {
            pendingSaveName.current = name;
            setSavePending(true);
            setSaveError(null);
            gw.racelineSave(name, lab.editorMap!, rl, lab.fastParams, lab.anchors);
          }}
          onClose={() => setShowSave(false)}
        />
      )}
    </div>
  );
}
