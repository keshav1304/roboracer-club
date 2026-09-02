"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { Gateway } from "@/lib/useGateway";
import type { Mapping } from "@/lib/useMapping";
import { isMapControllerAlgo } from "@/lib/algorithms";
import TrackCanvas, {
  type RacelineOverlay,
  type TrackMode,
  type TrackSource,
} from "@/components/TrackCanvas";
import VelocityChart from "@/components/VelocityChart";
import PoseErrorChart from "@/components/PoseErrorChart";

interface Props {
  gw: Gateway;
  lab: Mapping;
}

const ERR_HISTORY = 300;

/** Chooses what the central Mapping canvas shows for the active step. */
export default function MappingCanvas({ gw, lab }: Props) {
  const liveSeq = useRef(0);
  const lastFrame = useRef(gw.mapFrame);
  if (gw.mapFrame !== lastFrame.current) {
    lastFrame.current = gw.mapFrame;
    liveSeq.current += 1;
  }

  const [errXy, setErrXy] = useState<number[]>([]);
  const [errYaw, setErrYaw] = useState<number[]>([]);
  const racing = isMapControllerAlgo(gw.status.algorithm);

  // Accumulate true-vs-particle-filter pose error while racing; clear when not.
  useEffect(() => {
    if (!racing) {
      setErrXy([]);
      setErrYaw([]);
      return;
    }
    const pe = gw.telemetry.pose_error;
    if (!pe) return;
    setErrXy((prev) => [...prev, pe.xy_m].slice(-ERR_HISTORY));
    setErrYaw((prev) => [...prev, pe.yaw_deg].slice(-ERR_HISTORY));
  }, [racing, gw.telemetry.t, gw.telemetry.pose_error]);

  const savedSource = (name: string | null): TrackSource | null => {
    if (!name) return null;
    const meta = lab.maps.find((m) => m.name === name);
    if (meta) {
      return {
        key: `${name}-v${meta.version}`,
        gridUrl: lab.mapGridUrl(name, meta.version),
        url: lab.mapImageUrl(name, meta.version),
        width: meta.width,
        height: meta.height,
        resolution: meta.resolution,
        origin: [meta.origin[0] ?? 0, meta.origin[1] ?? 0],
        mapToWorld: meta.map_to_world ?? null,
        fitKey: name,
      };
    }
    // Parent map not in the list yet — still load the grid by name.
    return {
      key: `${name}-direct`,
      gridUrl: lab.mapGridUrl(name, 0),
      width: 1,
      height: 1,
      resolution: 0.05,
      origin: [0, 0],
      mapToWorld: null,
      fitKey: name,
    };
  };

  const liveSource: TrackSource | null = gw.mapFrame
    ? {
        key: `live-${liveSeq.current}`,
        occupancy: gw.mapFrame.occupancy,
        width: gw.mapFrame.width,
        height: gw.mapFrame.height,
        resolution: gw.mapFrame.resolution,
        origin: gw.mapFrame.origin,
        mapToWorld: gw.mapFrame.map_to_world,
        fitKey: `live-${Math.round(gw.mapFrame.width / 80)}x${Math.round(
          gw.mapFrame.height / 80
        )}`,
      }
    : null;

  const step = lab.step;
  const slamActive = gw.status.slam.active;

  let source: TrackSource | null = null;
  let mode: TrackMode = "view";
  let raceline: RacelineOverlay | null = null;
  let preview: { x: number[]; y: number[] } | null = null;
  let anchors: [number, number][] | undefined;
  let lookahead: { x: number; y: number } | null = null;
  let pathOverlay: { x: number; y: number }[] | null = null;
  let amclPose: { x: number; y: number; yaw: number } | null = null;
  let title = "Track";
  let subtitle = "Map frame";
  let meta = "";
  let emptyText = "Start mapping or open a saved map";
  const legend: { color: string; label: string; dashed?: boolean }[] = [
    { color: "#0ea36f", label: "Car (odom)" },
    { color: "rgba(59, 118, 232, 0.55)", label: "Trail" },
  ];

  if (step === "map") {
    source = slamActive || gw.mapFrame ? liveSource : savedSource(lab.viewedMap);
    title = "Map";
    subtitle = slamActive
      ? "Live SLAM"
      : lab.viewedMap
        ? `Saved · ${lab.viewedMap}`
        : "Live SLAM";
    if (!slamActive && lab.viewedMap && lab.tool !== "none") {
      mode = "edit-map";
      legend.push({
        color: lab.tool === "erase" ? "#0ea36f" : "#1c2333",
        label: lab.tool === "erase" ? "Erase brush" : "Wall brush",
      });
    }
    if (source) {
      meta = `${source.width}×${source.height} px`;
    }
  } else if (step === "raceline") {
    source = savedSource(lab.editorMap);
    title = "Generate Raceline";
    subtitle = lab.editorMap ? `Map · ${lab.editorMap}` : "";
    emptyText = "Pick a map in Generate Raceline";
    raceline = lab.raceline
      ? {
          x: lab.raceline.x,
          y: lab.raceline.y,
          v: lab.raceline.v,
          clearance: lab.raceline.clearance,
          margin: lab.raceline.min_clearance_m,
        }
      : null;
    if (lab.opt.running && lab.opt.preview) {
      preview = { x: lab.opt.preview.x, y: lab.opt.preview.y };
    }
    anchors = lab.anchors;
    mode = lab.raceline ? "edit-line" : "view";
    legend.push(
      { color: "hsl(220, 82%, 52%)", label: "Slow" },
      { color: "hsl(0, 82%, 52%)", label: "Fast" }
    );
    if (preview) {
      legend.push({ color: "#7c3aed", label: "Optimizer", dashed: true });
    }
  } else {
    // controller
    const overlay = lab.trackOverlay;
    source = savedSource(overlay?.meta.map ?? null);
    title = "Controller";
    subtitle = lab.selectedRaceline ? `Raceline · ${lab.selectedRaceline}` : "";
    emptyText = "Pick a raceline in the Controller step";
    if (overlay) {
      raceline = {
        x: overlay.data.x,
        y: overlay.data.y,
        v: overlay.data.v,
      };
      if (source) {
        meta = `${source.width}×${source.height} px`;
      }
    }
    const debug = gw.telemetry.algo_debug;
    if (debug && debug.algorithm === "pure_pursuit") {
      const tx = Number(debug.target_x);
      const ty = Number(debug.target_y);
      if (Number.isFinite(tx) && Number.isFinite(ty)) {
        lookahead = { x: tx, y: ty };
      }
    }
    if (debug && debug.algorithm === "mpc") {
      const path = debug.path;
      if (Array.isArray(path) && path.length > 1) {
        pathOverlay = path
          .map((pt) => {
            if (!Array.isArray(pt) || pt.length < 2) return null;
            const x = Number(pt[0]);
            const y = Number(pt[1]);
            if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
            return { x, y };
          })
          .filter((p): p is { x: number; y: number } => p != null);
      }
      const tx = Number(debug.target_x);
      const ty = Number(debug.target_y);
      if (Number.isFinite(tx) && Number.isFinite(ty)) {
        lookahead = { x: tx, y: ty };
      }
    }
    const pe = gw.telemetry.pose_error;
    if (pe?.amcl) {
      amclPose = pe.amcl;
    } else {
      const loc = gw.status.localize;
      if (loc.pose_fresh && loc.pose) {
        amclPose = {
          x: loc.pose.x,
          y: loc.pose.y,
          yaw: loc.pose.yaw,
        };
      }
    }
    legend.push(
      { color: "hsl(220, 82%, 52%)", label: "Slow" },
      { color: "hsl(0, 82%, 52%)", label: "Fast" }
    );
    if (debug?.algorithm === "mpc") {
      legend.push({ color: "#0891b2", label: "MPC horizon" });
      legend.push({ color: "#d97706", label: "Ref target", dashed: true });
    } else {
      legend.push({
        color: "#d97706",
        label: "Lookahead target",
        dashed: true,
      });
    }
    if (amclPose) {
      legend.push({ color: "#7c3aed", label: "Particle filter" });
    }
  }

  const sideExtra = useMemo(() => {
    if (step === "raceline" && lab.raceline) {
      return (
        <VelocityChart
          s={lab.raceline.s}
          v={lab.raceline.v}
          statsText={`${lab.raceline.length_m.toFixed(1)} m · est ${lab.raceline.lap_time_est.toFixed(1)} s`}
        />
      );
    }
    if (step === "controller" && lab.trackOverlay) {
      const debug = gw.telemetry.algo_debug;
      const carS =
        debug && isMapControllerAlgo(debug.algorithm)
          ? Number(debug.s_curr)
          : null;
      const trueSpeed = gw.telemetry.speed;
      const pe = gw.telemetry.pose_error;
      return (
        <>
          <VelocityChart
            s={lab.trackOverlay.data.s}
            v={lab.trackOverlay.data.v}
            carS={Number.isFinite(carS ?? NaN) ? carS : null}
            carSpeed={trueSpeed}
            statsText={
              lab.trackOverlay.meta.stats
                ? `est ${lab.trackOverlay.meta.stats.lap_time_est.toFixed(1)} s · ${trueSpeed.toFixed(2)} m/s`
                : `${trueSpeed.toFixed(2)} m/s odom`
            }
          />
          <PoseErrorChart
            xy={errXy}
            yawDeg={errYaw}
            latestXy={pe?.xy_m ?? null}
            latestYawDeg={pe?.yaw_deg ?? null}
          />
        </>
      );
    }
    return undefined;
  }, [
    step,
    lab.raceline,
    lab.trackOverlay,
    gw.telemetry.algo_debug,
    gw.telemetry.speed,
    gw.telemetry.pose_error,
    errXy,
    errYaw,
  ]);

  const track = (
    <TrackCanvas
      source={source}
      telemetry={gw.telemetry}
      trail={gw.trail}
      raceline={raceline}
      preview={preview}
      anchors={anchors}
      selectedAnchor={lab.selectedAnchor}
      lookahead={lookahead}
      pathOverlay={pathOverlay}
      amclPose={amclPose}
      mode={mode}
      tool={lab.tool}
      brushRadius={lab.brushRadius}
      onStroke={(points, tool, radius) => {
        if (lab.viewedMap) {
          gw.mapEdit(lab.viewedMap, [{ tool, points, radius }]);
        }
      }}
      onAnchorsChange={(pts) => lab.setAnchors(pts)}
      onAnchorSelect={lab.setSelectedAnchor}
      emptyText={emptyText}
      title={title}
      subtitle={subtitle}
      meta={meta}
      legend={legend}
      sideExtra={sideExtra}
      onClearTrail={gw.clearTrail}
    />
  );

  return track;
}
