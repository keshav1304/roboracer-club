"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { Gateway } from "@/lib/useGateway";
import type { Mapping } from "@/lib/useMapping";
import { suggestedMapName } from "@/lib/useMapping";
import { algorithmSpec } from "@/lib/algorithms";
import ParamsPanel from "@/components/ParamsPanel";
import NameModal from "./NameModal";
import SliderRow from "./SliderRow";

interface Props {
  gw: Gateway;
  lab: Mapping;
}

export default function MapStep({ gw, lab }: Props) {
  const { status } = gw;
  const slam = status.slam;
  const wallSpec = useMemo(() => algorithmSpec("wall_follow"), []);
  const [drive, setDrive] = useState<"wall_follow" | "manual">("wall_follow");
  const [wfParams, setWfParams] = useState<Record<string, number>>(() =>
    Object.fromEntries(wallSpec.params.map((p) => [p.name, p.default]))
  );
  const [showSave, setShowSave] = useState(false);
  const [savePending, setSavePending] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const lastAckN = useRef(0);

  // Resolve the save modal from slam save acks.
  useEffect(() => {
    const ack = gw.slamAck;
    if (!ack || ack.op !== "save" || ack.n === lastAckN.current) return;
    lastAckN.current = ack.n;
    setSavePending(false);
    if (ack.ok) {
      setShowSave(false);
      setSaveError(null);
      const saved = (ack as { map?: { name?: string } }).map;
      if (saved?.name) lab.setViewedMap(saved.name);
    } else {
      setSaveError(ack.reason ?? "Save failed");
    }
  }, [gw.slamAck, lab]);

  const canStart = status.wsConnected && status.simConnected && !slam.active;
  const editing =
    !slam.active && lab.viewedMap != null && lab.tool !== "none";
  const wallFollowMapping =
    slam.active && slam.drive === "wall_follow";
  const showWallParams = !slam.active
    ? drive === "wall_follow"
    : slam.drive === "wall_follow";

  return (
    <div className="step-body">
      {!slam.active ? (
        <section className="step-section">
          <div className="field-label">Drive mode</div>
          <select
            className="select"
            value={drive}
            onChange={(e) =>
              setDrive(e.target.value as "wall_follow" | "manual")
            }
          >
            <option value="wall_follow">Wall follow</option>
            <option value="manual">Manual</option>
          </select>
          <div className="btn-row">
            <button
              className="btn primary"
              disabled={!canStart}
              onClick={() => {
                lab.setViewedMap(null);
                gw.slamStart(
                  drive,
                  drive === "wall_follow" ? wfParams : undefined
                );
              }}
            >
              Start mapping
            </button>
          </div>
          {!status.simConnected && (
            <div className="note-inline">
              Connect the simulator first (Connection panel).
            </div>
          )}
          {drive === "manual" ? (
            <div className="note-inline">
              Set the simulator to Manual. Drive a few slow laps, then save.
            </div>
          ) : (
            <div className="note-inline">
              Set the simulator to Autonomous. Adjust wall follow below, then
              map for a few laps before saving.
            </div>
          )}
        </section>
      ) : (
        <section className="step-section">
          <div className="status-row">
            <span className="dot on" />
            Mapping
            <span className="hint">
              {slam.drive === "wall_follow" ? "wall follow" : "manual"}
            </span>
          </div>
          <div className="btn-row">
            <button
              className="btn primary"
              onClick={() => {
                setSaveError(null);
                setShowSave(true);
              }}
            >
              Save map
            </button>
            <button className="btn danger" onClick={() => gw.slamStop()}>
              Stop
            </button>
          </div>
        </section>
      )}

      {showWallParams && (
        <section className="step-section">
          <ParamsPanel
            algorithm={wallSpec}
            onSetParams={(p) => {
              setWfParams((prev) => ({ ...prev, ...p }));
              if (wallFollowMapping) gw.setParams(p);
            }}
            lastAck={wallFollowMapping ? gw.lastAck : null}
            disabled={false}
          />
        </section>
      )}

      {!slam.active && lab.viewedMap && (
        <section className="step-section">
          <div className="field-label">
            Cleanup <span className="mono">{lab.viewedMap}</span>
          </div>
          <div className="btn-row">
            <button
              className={`btn small ${lab.tool === "erase" ? "primary" : ""}`}
              onClick={() =>
                lab.setTool(lab.tool === "erase" ? "none" : "erase")
              }
            >
              Erase
            </button>
            <button
              className={`btn small ${lab.tool === "wall" ? "primary" : ""}`}
              onClick={() => lab.setTool(lab.tool === "wall" ? "none" : "wall")}
            >
              Draw wall
            </button>
            <button
              className="btn small"
              onClick={() => gw.mapUndo(lab.viewedMap!)}
            >
              Undo
            </button>
            <button
              className="btn small"
              onClick={() => gw.mapRedo(lab.viewedMap!)}
            >
              Redo
            </button>
          </div>
          {editing && (
            <div className="step-block">
              <SliderRow
                label="Brush size (px)"
                min={1}
                max={12}
                step={1}
                value={lab.brushRadius}
                onChange={lab.setBrushRadius}
              />
              <div className="note-inline">
                {lab.tool === "erase"
                  ? "Paint to clear noise and stray walls."
                  : "Paint to fill gaps in walls."}{" "}
                Shift-drag to pan.
              </div>
            </div>
          )}
          {gw.mapAck && !gw.mapAck.ok && (
            <div className="ack fail">{gw.mapAck.reason}</div>
          )}
        </section>
      )}

      <section className="step-section">
        <div className="field-label">Saved maps</div>
        {lab.maps.length === 0 ? (
          <div className="note-inline">No maps yet. Map a lap and save it.</div>
        ) : (
          <div className="artifact-list">
            {lab.maps.map((m) => (
              <div
                key={m.name}
                className={`artifact ${lab.viewedMap === m.name ? "active" : ""}`}
              >
                <img
                  className="artifact-thumb"
                  src={lab.mapImageUrl(m.name, m.version)}
                  alt=""
                />
                <div className="artifact-info">
                  <div className="artifact-name">{m.name}</div>
                  <div className="artifact-sub">
                    {m.width}×{m.height} px · {m.resolution} m/px
                  </div>
                </div>
                <div className="artifact-actions">
                  <button
                    className="btn small"
                    onClick={() =>
                      lab.setViewedMap(lab.viewedMap === m.name ? null : m.name)
                    }
                  >
                    {lab.viewedMap === m.name ? "Close" : "Open"}
                  </button>
                  <button
                    className="btn small"
                    onClick={() => {
                      lab.setEditorMap(m.name);
                      lab.setStep("raceline");
                    }}
                  >
                    Raceline
                  </button>
                  <button
                    className="btn small danger"
                    onClick={() => {
                      if (confirm(`Delete map "${m.name}"?`)) {
                        if (lab.viewedMap === m.name) lab.setViewedMap(null);
                        gw.mapDelete(m.name);
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

      {showSave && (
        <NameModal
          title="Save map"
          description="Saves the live SLAM map so you can build a raceline on it."
          suggestion={suggestedMapName()}
          existingNames={lab.maps.map((m) => m.name)}
          busy={savePending}
          error={saveError}
          onSave={(name) => {
            setSavePending(true);
            setSaveError(null);
            gw.slamSaveMap(name);
          }}
          onClose={() => setShowSave(false)}
        />
      )}
    </div>
  );
}
